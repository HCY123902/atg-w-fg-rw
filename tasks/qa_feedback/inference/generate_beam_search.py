import sys
import os
import shutil

from fgrlhf.reward_utils import RewardModelBeamSearch

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
parser.add_argument("--base_model_name_or_path", type=str, help="Name or path of the base model checkpoint", default="meta-llama/Llama-2-7b-hf")
# parser.add_argument("--save_path", required=True, type=str, help="Path to save the evaluation result")
parser.add_argument("--dataset", required=True, type=str, help="Evaluation dataset", choices=["asqa", "qampari", "eli5", "combined", "expertqa"])
parser.add_argument("--num_return_sequences", type=int, default=2, help="Number of return sequences")
parser.add_argument("--top_k", type=int, default=50, help="Top K for sampling")
parser.add_argument("--top_p", type=float, default=1.0, help="Top P for sampling")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperate for sampling")
parser.add_argument("--load_from_last_pth", help="Whether to load from the checkpoint saved with the save_pretrained method", action="store_true")
parser.add_argument("--eval_step", type=int, help="Training step of the evaluated checkpoint", default=-1)
parser.add_argument("--use_base", help="Whether to use base model to evaluate directly", action="store_true")
parser.add_argument("--include_demo", action="store_true", help="Whether or not to include in context demonstrations")
# parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
parser.add_argument("--start_idx", type=int, default=0, help="Starting sample")
parser.add_argument("--beam_size", type=int, default=2, help="The number of beams to use")
parser.add_argument("--max_depth", type=int, default=5, help="The maximum search depth in terms of the number of segments")
parser.add_argument("--inference", action="store_true", help="Whether or not to use the sampled result for direct evaluation instead of another round of training")
parser.add_argument("--inference_save_path", type=str, help="The path to save inference results", default="./tasks/qa_feedback/model_outputs/rs-fg-direct-inference")
parser.add_argument("--init_prob", type=float, default=-1, help="Initial probability for random tansition")
parser.add_argument("--no_citation_reward", action="store_true", help="Whether or not to remove citation reward")
parser.add_argument("--no_correctness_reward", action="store_true", help="Whether or not to remove correctness reward")

args = parser.parse_args()

dataset = args.dataset

if dataset != "qampari":
    assert args.init_prob == -1
else:
    assert args.init_prob == -1 or (args.init_prob > 0 and args.init_prob <= 1)

peft_ckpt = args.peft_ckpt

base_model_path=args.base_model_name_or_path

# Since the same tokenizer is used in different settings, tokenizer_path can point to any valid copy of tokenizer, even from different checkpoints
tokenizer_path = "./tasks/qa_feedback/model_outputs/distillation" if args.use_base else args.peft_ckpt

assert args.use_base == args.include_demo

if not args.use_base:
    eval_step_suffix = "_ckp_{}".format(args.eval_step) if args.eval_step >= 0 else ""

    merged_model_path = os.path.join(peft_ckpt, "merged_with_root_peft{}".format(eval_step_suffix))
    # merged_model_path = base_model_path

    # logger.info("-----------------------{}-----------------------".format(args.load_from_last_pth))

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



model_path = base_model_path if args.use_base else merged_model_path

stop_tokens=["\n", "Ċ", "ĊĊ", "<0x0A>", "</s>", "."] if dataset != "qampari" else ["\n", "Ċ", "ĊĊ", "<0x0A>", "</s>", ","]

device = 0 if (args.inference or dataset != "eli5") and args.beam_size <= 2 else 1
# Run with init sft ckpt for evaluation
vllm = VLLM(model_path, tokenizer_name_or_path=tokenizer_path, sample=True, num_return_sequences=args.num_return_sequences, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, stop_tokens=stop_tokens, max_model_len=4096, gpu_memory_utilization=0.37 if device == 0 else 0.9)

init_autoais = dataset in ["eli5", "expertqa", "combined"] or not args.no_citation_reward

# print(init_autoais, dataset in ["eli5", "expertqa", "combined"], args.no_citation_reward)

beam_search_rw_m = RewardModelBeamSearch(autoais_model_name="google/t5_xxl_true_nli_mixture", autoais_model_type="bf16", device=device, inference=args.inference, no_citation_reward=args.no_citation_reward, no_correctness_reward=args.no_correctness_reward, init_autoais=init_autoais)

vllm.init_beam_search_rw_m(beam_search_rw_m)

if args.inference:
    eval_json_path = "./tasks/qa_feedback/data/{}_test.json".format(dataset)
else:
    eval_json_path = "./tasks/qa_feedback/data/{}_train.json".format(dataset)

with open(eval_json_path, "r") as eval_json:
    eval_samples = json.load(eval_json)[args.start_idx:]

save_path = args.inference_save_path

if args.inference and not os.path.exists(save_path):
    os.mkdir(save_path)

save_steps = 100

def get_res_save_path(save_path, dataset, retriever, shot, eval_json_path, args, step=-1):
    start_idx_suffix = "_{}".format(args.start_idx) if args.start_idx != 0 else ""
    # init_prob_suffix = "_init_prob_{}".format(args.init_prob) if args.init_prob > 0 else ""
    no_citation_reward_suffix = "_no_citation_reward" if args.no_citation_reward else ""
    no_correctness_reward_suffix = "_no_correctness_reward" if args.no_correctness_reward else ""

    if args.inference:
        if args.use_base:
            res_save_path = os.path.join(save_path, "{}-rlhf-test-vllm-llama-2-7b-hf-{}-shot{}-ndoc5-42-constrained-decoding{}{}{}.json".format(dataset, retriever, shot, start_idx_suffix, no_citation_reward_suffix, no_correctness_reward_suffix))
        else:
            res_save_path = os.path.join(save_path, "{}_result{}.json".format(dataset, start_idx_suffix))
    else:
        # demo_suffix = "_shot_2" if args.include_demo else ""
        res_save_path = eval_json_path.replace(".json", "_rs_fg.json")
    if step == save_steps - 1 and os.path.exists(res_save_path):
        raise Exception("{} already exists".format(res_save_path))
    return res_save_path

def save_samples(eval_samples, dataset, save_path, args, step=-1):
    if args.inference:
        for s in eval_samples:
            s["output"] = s["output"][0]
            s["gpt_output"] = s["gpt_output"][0]
        if dataset == "combined":
            for name in ["asqa", "qampari", "eli5"]:
                samples = [s for s in eval_samples if s["dataset"] == name]
                res_save_path = get_res_save_path(save_path, name, "gtr" if name != "eli5" else "bm25", 2 if args.include_demo else 0, eval_json_path, args, step=step)
                with open(res_save_path, "w") as samples_json:
                    json.dump({"data": samples}, samples_json, indent=4)
        else:
            res_save_path = get_res_save_path(save_path, dataset, "gtr" if dataset != "eli5" else "bm25", 2 if args.include_demo else 0, eval_json_path, args, step=step)
            with open(res_save_path, "w") as eval_samples_json:
                json.dump({"data": eval_samples}, eval_samples_json, indent=4)
    else:
        res_save_path = get_res_save_path(save_path, dataset, "gtr" if dataset != "eli5" else "bm25", 2 if args.include_demo else 0, eval_json_path, args, step=step)
        with open(res_save_path, "w") as eval_samples_json:
                json.dump(eval_samples, eval_samples_json, indent=4)

for step in tqdm(range(len(eval_samples))):
    item = eval_samples[step]
    if "chat" in base_model_path:
        raise Exception("Not implemented")
    time_start = time.time()
    temp = item["output"][0] if len(item["output"]) > 0 else ""

    for d in range(1, args.max_depth+1):
        item["beam_search_depth_{}".format(d)] = []

    curr_seqs = vllm.generate_beam_search(item, beam_size=args.beam_size, max_depth=args.max_depth, include_demo=args.include_demo, init_prob=args.init_prob)
    logger.info("Time taken: {:2f} seconds".format(time.time() - time_start))
    logger.info("================================")

    item["beam_search"] = curr_seqs
    item["output"] = [curr_seqs[0]["pred_text"]]
    item["gpt_output"] = [temp]

    logger.info("Question: {}".format(item["question"]))
    logger.info("Answer: {}".format(curr_seqs[0]["pred_text"]))


    if (step + 1) % save_steps == 0:
        logger.info("Saving samples at step {}".format(step))
        save_samples(eval_samples[:step+1], dataset, save_path, args, step=step)

save_samples(eval_samples, dataset, save_path, args)

if not args.use_base:
    logging.info("Removing the merged checkpoint")
    shutil.rmtree(merged_model_path)