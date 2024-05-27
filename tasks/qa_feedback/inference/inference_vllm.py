import sys
import os
import shutil

log_file_path = os.environ["RUN_LOG_PATH"]

log_fh = open(log_file_path, 'a')

sys.stderr = log_fh
sys.stdout = log_fh

import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler(log_file_path, mode="a"),
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import json
import time

from tqdm import tqdm

from fgrlhf.utils import *

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

from peft import PeftConfig, PeftModel, get_peft_model, set_peft_model_state_dict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--peft_ckpt", type=str, help="Path to policy_adapter if not loading from last checkpoint and path to model root save path otherwise", default=None)
parser.add_argument("--tokenizer_path", type=str, help="Path to the init sft tokenizer", default=None)
parser.add_argument("--base_model_name_or_path", type=str, help="Name or path of the base model checkpoint", default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--save_path", required=True, type=str, help="Path to save the evaluation result")
parser.add_argument("--dataset", required=True, type=str, help="Evaluation dataset", choices=["asqa", "qampari", "eli5", "combined", "expertqa"])
parser.add_argument("--load_from_last_pth", help="Whether to load from the checkpoint saved with the save_pretrained method", action="store_true")
parser.add_argument("--eval_step", type=int, help="Training step of the evaluated checkpoint", default=-1)
parser.add_argument("--use_base", help="Whether to use base model to evaluate directly", action="store_true")
parser.add_argument("--include_demo", help="Whether or not to include in context demonstrations", action="store_true")
parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")

args = parser.parse_args()

dataset = args.dataset


peft_ckpt = args.peft_ckpt

# Since the same tokenizer is used in different settings, tokenizer_path can point to any valid copy of tokenizer, even from different checkpoints
tokenizer_path = args.tokenizer_path

base_model_path=args.base_model_name_or_path


if not args.use_base:
    eval_step_suffix = "_ckp_{}".format(args.eval_step) if args.eval_step >= 0 else ""

    merged_model_path = os.path.join(peft_ckpt, "merged_with_root_peft{}".format(eval_step_suffix))

    load_from_last_pth = args.load_from_last_pth

    if not os.path.exists(merged_model_path) or "config.json" not in os.listdir(merged_model_path):
        logger.info("Merging and saving the model with the tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=1200)

        tokenizer.max_input_len = 1200
        tokenizer.max_generated_len = 200

        base_model = load_model(base_model_path, tokenizer, init_sft=False, inference=True, quantize=False)

        if not load_from_last_pth:
            peft_config = PeftConfig.from_pretrained(peft_ckpt)
            peft_model = PeftModel.from_pretrained(base_model, peft_ckpt, is_trainable=True)
        else:
            # Source: https://github.com/huggingface/peft/blob/main/src/peft/utils/save_and_load.py#L157
            peft_config = get_qlora_config()
            # peft_model = base_model
            # peft_model.add_adapter("policy_adapter", peft_config)
            peft_model = get_peft_model(base_model, peft_config)
            adapter_weights = torch.load(os.path.join(peft_ckpt, "last{}.pth".format(eval_step_suffix)), map_location="cpu")["model"]
            # set_peft_model_state_dict(peft_model, adapter_weights, adapter_name="policy_adapter")
            set_peft_model_state_dict(peft_model, adapter_weights, adapter_name="default")

        peft_merge_and_save(peft_model, merged_model_path, tokenizer=tokenizer)

        del base_model
        del peft_model

        assert tokenizer.eos_token == "</s>"

logger.info("Running VLLM inference")



# Run with init sft ckpt for evaluation
if args.use_base:
    vllm = VLLM(base_model_path, tokenizer_name_or_path=args.tokenizer_path, sample=False, max_model_len=4096)
else:
    vllm = VLLM(merged_model_path, sample=False, max_model_len=4096, gpu_memory_utilization=0.5)

eval_json_path = "./tasks/qa_feedback/data/{}_test.json".format(dataset)

with open(eval_json_path, "r") as eval_json:
    eval_samples = json.load(eval_json)

batch_size = args.batch_size

for i in tqdm(range(0, len(eval_samples), batch_size)):
    if "chat" in base_model_path:
        if args.include_demo:
            raise Exception("Not implemented")
        prompts = [get_chat_prompt_from_promt_without_demo(prompt_without_demo=sample['prompt_without_demo']) for sample in eval_samples[i:i+batch_size]]
    else:
        if args.include_demo:
            prompts = [sample["prompt"] for sample in eval_samples[i:i+batch_size]]
        else:
            prompts = [sample["prompt_without_demo"] for sample in eval_samples[i:i+batch_size]]
    time_start = time.time()
    generated_texts, generated_input_ids = vllm.generate(prompts)
    logger.info("Tokens: {} | Time taken: {:2f} seconds".format(sum([len(input_ids) for input_ids in generated_input_ids]), time.time() - time_start))
    for j in range(min(batch_size, len(eval_samples) - i)):
        logger.info("================{} {}================".format(i, j))
        temp = eval_samples[i+j]["output"][0] if len(eval_samples[i+j]["output"]) > 0 else ""
        eval_samples[i+j]["output"] = generated_texts[j]
        eval_samples[i+j]["gpt_output"] = temp

        logger.info("Question: {}".format(eval_samples[i+j]["question"]))
        logger.info("Answer: {}".format(generated_texts[j]))

save_path = args.save_path

if not os.path.exists(save_path):
    os.mkdir(save_path)

if dataset != "combined":
    if args.use_base:
        res_save_path = os.path.join(save_path, "{}-rlhf-test-vllm-llama-2-7b-hf-{}-shot{}-ndoc5-42.json".format(dataset, "gtr" if dataset not in ["eli5", "expertqa"] else "bm25", 2 if args.include_demo else 0))
    else:
        res_save_path = os.path.join(save_path, "{}_result.json".format(dataset))
    with open(res_save_path, "w") as eval_samples_json:
        json.dump({"data": eval_samples}, eval_samples_json, indent=4)
else:
    for name in ["asqa", "qampari", "eli5"]:
        samples = [s for s in eval_samples if s["dataset"] == name]
        if args.use_base:
            res_save_path = os.path.join(save_path, "{}-rlhf-test-vllm-llama-2-7b-hf-{}-shot{}-ndoc5-42.json".format(name, "gtr" if name not in ["eli5", "expertqa"] else "bm25", 2 if args.include_demo else 0))
        else:
            res_save_path = os.path.join(save_path, "{}_result.json".format(name))
        with open(res_save_path, "w") as samples_json:
            json.dump({"data": samples}, samples_json, indent=4)

if not args.use_base:
    logging.info("Removing the merged checkpoint")
    shutil.rmtree(merged_model_path)
