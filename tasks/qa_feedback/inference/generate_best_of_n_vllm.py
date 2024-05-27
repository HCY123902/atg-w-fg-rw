from ast import parse
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
from fgrlhf.reward_utils import RewardModelBestN

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

import time
import gc

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--peft_ckpt", type=str, help="Path to policy_adapter if not loading from last checkpoint and path to model root save path otherwise", default=None)
parser.add_argument("--base_model_name_or_path", type=str, help="Name or path of the base model checkpoint", default="meta-llama/Llama-2-7b-hf")
# parser.add_argument("--save_path", required=True, type=str, help="Path to save the evaluation result")
parser.add_argument("--dataset", required=True, type=str, help="Evaluation dataset", choices=["asqa", "qampari", "eli5", "combined", "expertqa"])
parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of return sequences")
parser.add_argument("--top_k", type=int, default=50, help="Top K for sampling")
parser.add_argument("--top_p", type=float, default=1.0, help="Top P for sampling")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperate for sampling")
parser.add_argument("--use_base", help="Whether to use base model to evaluate directly", action="store_true")
parser.add_argument("--include_demo", action="store_true", help="Whether or not to include in context demonstrations")
parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
parser.add_argument("--start_idx", type=int, default=0, help="Starting sample")
parser.add_argument("--inference", action="store_true", help="Whether or not to use the sampled result for direct evaluation instead of another round of training")
parser.add_argument("--inference_save_path", type=str, help="The path to save inference results", default="./tasks/qa_feedback/model_outputs/rs-h-direct-inference")

args = parser.parse_args()

dataset = args.dataset

peft_ckpt = args.peft_ckpt

base_model_path=args.base_model_name_or_path

# Since the same tokenizer is used in different settings, tokenizer_path can point to any valid copy of tokenizer, even from different checkpoints
tokenizer_path = "./tasks/qa_feedback/model_outputs/distillation" if args.use_base else args.peft_ckpt

assert args.use_base == args.include_demo

if not args.use_base:

    merged_model_path = os.path.join(peft_ckpt, "merged_with_root_peft")
    # merged_model_path = base_model_path

    # logger.info("-----------------------{}-----------------------".format(args.load_from_last_pth))

    if not os.path.exists(merged_model_path) or "config.json" not in os.listdir(merged_model_path):
        logger.info("Merging and saving the model with the tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=1200)

        tokenizer.max_input_len = 1200
        tokenizer.max_generated_len = 200

        base_model = load_model(base_model_path, tokenizer, init_sft=False, inference=True, quantize=False)

        peft_config = PeftConfig.from_pretrained(peft_ckpt)
        peft_model = PeftModel.from_pretrained(base_model, peft_ckpt, is_trainable=True)

        peft_merge_and_save(peft_model, merged_model_path, tokenizer=tokenizer)

        del base_model
        del peft_model

        assert tokenizer.eos_token == "</s>"

logger.info("Running VLLM inference")



num_return_sequences = args.num_return_sequences

if args.inference:
    eval_json_path = "./tasks/qa_feedback/data/{}_test.json".format(dataset)
else:
    eval_json_path = "./tasks/qa_feedback/data/{}_train.json".format(dataset)

with open(eval_json_path, "r") as eval_json:
    eval_samples = json.load(eval_json)[args.start_idx:]

demo_suffix = "_shot_2" if args.include_demo else ""
start_idx_suffix = "_{}" if args.start_idx > 0 else ""

if args.include_demo:
    logger.info("Including in context demonstrations in the prompt")

if args.inference:
    assert not args.use_base
    save_root_dir = args.inference_save_path
    if not os.path.exists(save_root_dir):
        os.mkdir(save_root_dir)
    temp_path = os.path.join(save_root_dir, "{}_result_temp.json".format(dataset))
    save_path = os.path.join(save_root_dir, "{}_result.json".format(dataset))
else:
    temp_path = eval_json_path.replace(".json", "_rs_h_temp.json")
    save_path = eval_json_path.replace(".json", "_rs_h.json")

if not os.path.exists(temp_path):
    batch_size = args.batch_size

    total_generated_texts = []

    model_path = base_model_path if args.use_base else merged_model_path

    vllm = VLLM(model_path, tokenizer_name_or_path=tokenizer_path, sample=True, num_return_sequences=num_return_sequences, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, max_model_len=4096)
        

    logger.info("Sampling the generated text")
    for i in tqdm(range(0, len(eval_samples), batch_size)):
        if "chat" in model_path:
            if args.include_demo:
                raise Exception("Not implemented")
            prompts = [get_chat_prompt_from_promt_without_demo(prompt_without_demo=sample['prompt_without_demo']) for sample in eval_samples[i:i+batch_size]]
        else:
            if args.include_demo:
                prompts = [sample["prompt"] for sample in eval_samples[i:i+batch_size]]
            else:
                prompts = [sample["prompt_without_demo"] for sample in eval_samples[i:i+batch_size]]
        time_start = time.time()
        try:
            generated_texts, generated_input_ids = vllm.generate(prompts)
        except Exception as e:
            logger.info("Encountered {}. Will initialized vllm and then generate again".format(e))
            del vllm
            time.sleep(30)
            vllm = VLLM(model_path, tokenizer_name_or_path=tokenizer_path, sample=True, num_return_sequences=num_return_sequences, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, max_model_len=4096)
            generated_texts, generated_input_ids = vllm.generate(prompts)
        logger.info("Tokens: {} | Time taken: {:2f} seconds".format(sum([len(input_ids) for input_ids in generated_input_ids]), time.time() - time_start))
        for j in range(min(batch_size, len(eval_samples) - i)):
            logger.info("================{} {}================".format(i, j))
            curr_generated_texts = generated_texts[j*num_return_sequences:(j+1)*num_return_sequences]
            total_generated_texts.append(curr_generated_texts)
            # eval_samples[i+j]["output"] = generated_texts[j]

            logger.info("Question: {}".format(eval_samples[i+j]["question"]))
            logger.info("Gold: {}".format(eval_samples[i+j]["output"][0]))
            logger.info("Answer: {}".format(curr_generated_texts))

    # save_path = args.save_path

    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

    # if dataset != "combined":
    #     with open(os.path.join(save_path, "{}_result.json".format(dataset)), "w") as eval_samples_json:
    #         json.dump({"data": eval_samples}, eval_samples_json, indent=4)
    # else:
    #     for name in ["asqa", "qampari", "eli5"]:
    #         samples = [s for s in eval_samples if s["dataset"] == name]
    #         with open(os.path.join(save_path, "{}_result.json".format(name)), "w") as samples_json:
    #             json.dump({"data": samples}, samples_json, indent=4)

    with open(temp_path, "w") as temp_json:
        json.dump(total_generated_texts, temp_json, indent=4)

    del vllm

    torch.cuda.empty_cache()
    gc.collect()

    time.sleep(30)

else:
    logger.info("Directly loading the generated texts from {}".format(temp_path))
    with open(temp_path, 'r') as f:
        total_generated_texts = json.load(f)

logger.info("Computing the rewards and selecting the output for the samples")

rw_model = RewardModelBestN(autoais_model_name="google/t5_xxl_true_nli_mixture", autoais_model_type="bf16", inference=args.inference)

rw_model.get_reward(eval_samples, total_generated_texts, num_return_sequences, args.dataset)


if args.inference:
    for s in eval_samples:
        s["output"] = s["output"][0]
        s["gpt_output"] = s["gpt_output"][0]
    if dataset == "combined":
        for name in ["asqa", "qampari", "eli5"]:
            samples = [s for s in eval_samples if s["dataset"] == name]
            with open(save_path.replace("combined_result.json", "{}_result.json".format(name)), "w") as samples_json:
                json.dump({"data": samples}, samples_json, indent=4)
    else:
        with open(save_path, "w") as tgt_json:
            json.dump({"data": eval_samples}, tgt_json, indent=4)
else:
    with open(save_path, "w") as tgt_json:
        json.dump(eval_samples, tgt_json, indent=4)

if not args.use_base:
    logging.info("Removing the merged checkpoint")
    shutil.rmtree(merged_model_path)
