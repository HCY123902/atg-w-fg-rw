import os
import sys

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

from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

from fgrlhf.reward_utils import RewardModelBestN

import argparse

import time
import gc

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, help="Name or path of the base model checkpoint", default="t5-large")
parser.add_argument("--dataset", required=True, type=str, help="Evaluation dataset", choices=["asqa", "qampari", "eli5", "combined", "expertqa"])
parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of return sequences")
parser.add_argument("--top_k", type=int, default=50, help="Top K for sampling")
parser.add_argument("--top_p", type=float, default=1.0, help="Top P for sampling")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperate for sampling")
parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
parser.add_argument("--inference", action="store_true", help="Whether or not to use the sampled result for direct evaluation instead of another round of training")
parser.add_argument("--inference_save_path", type=str, help="The path to save inference results", default="./tasks/qa_feedback/model_outputs/rs-h-direct-inference")

args = parser.parse_args()

class TextGenDatasetBestN(Dataset):
    def __init__(self, split, tokenizer, accelerator=None, length_limit=None, dataset_name=None, model_type="t5"):
        super().__init__()
        
        self.split = split

        dataset_suffix = "shot_2_ndoc_5_ndoc_in_demo_5_default_inst" if model_type == "llama" else "shot_1_ndoc_5_ndoc_in_demo_5_no_inst"

        self.dataset_map = {
            "asqa": {
                "train": "./tasks/qa_feedback/data/asqa_rlhf_train_{}.json".format(dataset_suffix),
                "dev": "./tasks/qa_feedback/data/asqa_rlhf_dev_{}.json".format(dataset_suffix),
                "test": "./tasks/qa_feedback/data/asqa_rlhf_test_{}.json".format(dataset_suffix)
            },
            "qampari": {
                "train": "./tasks/qa_feedback/data/qampari_rlhf_train_{}.json".format(dataset_suffix),
                "dev": "./tasks/qa_feedback/data/qampari_rlhf_dev_{}.json".format(dataset_suffix),
                "test": "./tasks/qa_feedback/data/qampari_rlhf_test_{}.json".format(dataset_suffix)
            },
            "eli5": {
                "train": "./tasks/qa_feedback/data/eli5_rlhf_train_{}.json".format(dataset_suffix),
                "dev": "./tasks/qa_feedback/data/eli5_rlhf_dev_{}.json".format(dataset_suffix),
                "test": "./tasks/qa_feedback/data/eli5_rlhf_test_1000_{}.json".format(dataset_suffix)
            },
            "combined": {
                "train": "./tasks/qa_feedback/data/combined_rlhf_train_{}.json".format(dataset_suffix),
                "dev": "./tasks/qa_feedback/data/combined_rlhf_dev_{}.json".format(dataset_suffix),
                "test": "./tasks/qa_feedback/data/combined_rlhf_test_{}.json".format(dataset_suffix)
            }
        }

        self.dataset_fns = self.dataset_map[dataset_name]

        logger.info(f'Currently loading split {self.split} from {dataset_name}')
        
        self.n_card = 1
        if accelerator is not None:
            self.n_card = accelerator.num_processes
        
        
        self.tokenizer = tokenizer

        self.instances = self.load_datasets()
        
        if length_limit is not None:
            self.instances = self.instances[:length_limit]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self): 
        instances = []
        
        task_data = None
        with open(self.dataset_fns[self.split], 'r') as f:
            task_data = json.load(f)
            
        for task_instance in task_data:

            human_answer = [task_instance['answer']] if task_instance['dataset'] == "eli5" else ([annot["long_answer"] for annot in task_instance["annotations"][:2]] if task_instance['dataset'] == "asqa" else []) # MAUVE or Rouge; Empty for qampari

            instances.append({
                "prompt": task_instance['prompt_without_demo'],
                "metadata": {
                    "prompt": task_instance['prompt_without_demo'],
                    # "references": task_instance['answer'],
                    "human_answer": human_answer,
                    "docs": task_instance['docs'], # citation recall and precision
                    "question": task_instance['question'],
                    "dataset": task_instance['dataset'],
                    "answers": task_instance['answers'], # qampari correctness
                    "qa_pairs": task_instance['qa_pairs'], # ASQA correctness
                    "claims": task_instance['claims'], # ELI5 correctness
                    # "doc_rec_ans": task_instance['doc_rec_ans'],
                },

            })
        
        logger.info(f'Loaded split {self.split} with {len(instances)} total instances')
        
        instances = instances[:len(instances)//self.n_card*self.n_card]  # or Trainer will stuck
        return instances

    # Make a collate function to fix dataloader weird list batching
    def collate_fn(self, batch):
        
        # process input prompts
        prompts = [item['prompt'] for item in batch]
        prompts_tok = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors='pt', 
            padding='max_length', # Will need to change it if demonstrations are included in the prompt. The default instruction already has 107 tokens.
            truncation=True,
            max_length=self.tokenizer.max_input_len,
            # padding_side=self.tokenizer.padding_side, # YUSHI: change later, now Ellen pad defaultly
            )
        
        prompts_input_ids = prompts_tok.input_ids
        prompts_attention_mask = prompts_tok.attention_mask
        
        # process metadata
        metadata = [item['metadata'] for item in batch]
        

        result = {
            'prompts_input_ids': prompts_input_ids,
            'prompts_attention_mask': prompts_attention_mask,
            'metadata': metadata
        }
        return result

model_name_or_path = args.model_name_or_path
model_type = "t5"
dataset = args.dataset

cache_dir = None
revision = "main"

num_return_sequences = args.num_return_sequences

config = AutoConfig.from_pretrained(
    model_name_or_path,
    cache_dir=cache_dir,
    revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    cache_dir=cache_dir,
    use_fast=True,
    revision=revision
)
tokenizer.max_input_len = 1024
tokenizer.max_generated_len = 200

if args.inference:
    eval_dataset = TextGenDatasetBestN('test', tokenizer, accelerator=None, dataset_name=dataset, model_type=model_type)
else:
    eval_dataset = TextGenDatasetBestN('train', tokenizer, accelerator=None, dataset_name=dataset, model_type=model_type)

eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, 
                                shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)
if args.inference:
    assert "tasks" in model_name_or_path
    save_root_dir = args.inference_save_path
    if not os.path.exists(save_root_dir):
        os.mkdir(save_root_dir)
    temp_path = os.path.join(save_root_dir, "{}_result_temp.json".format(dataset))
    save_path = os.path.join(save_root_dir, "{}_result.json".format(dataset))
else:
    temp_path = eval_dataset.dataset_fns[eval_dataset.split].replace(".json", "_rs_h.json")
    save_path = eval_dataset.dataset_fns[eval_dataset.split].replace(".json", "_rs_h.json")

if not os.path.exists(temp_path):

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir,
        revision=revision,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    total_generated_texts = []

    logger.info("Sampling the generated text")
    for batch in tqdm(eval_dataloader):
        generated_input_ids = model.generate(
            input_ids=batch["prompts_input_ids"].to(model.device),
            attention_mask=batch["prompts_attention_mask"].to(model.device),
            max_length=tokenizer.max_generated_len + 1,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_return_sequences=num_return_sequences,
            # synced_gpus=self.synced_gpus,
        ) # begins with 0 ([BOS]); ends with 1 ([EOS])

        generated_texts = tokenizer.batch_decode(generated_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        assert len(generated_texts) == len(batch["prompts_input_ids"]) * num_return_sequences

        for i in range(0, len(generated_texts), num_return_sequences):
            total_generated_texts.append(generated_texts[i:i+num_return_sequences])

    with open(temp_path, "w") as temp_json:
        json.dump(total_generated_texts, temp_json, indent=4)

    del model

    torch.cuda.empty_cache()
    gc.collect()

    time.sleep(30)

else:
    logger.info("Directly loading the generated texts from {}".format(temp_path))
    with open(temp_path, 'r') as f:
        total_generated_texts = json.load(f)


logger.info("Computing the rewards and selecting the output for the samples")

with open(eval_dataset.dataset_fns[eval_dataset.split], 'r') as f:
    data = json.load(f)

rw_model = RewardModelBestN(autoais_model_name="google/t5_xxl_true_nli_mixture", autoais_model_type="bf16", inference=args.inference)

rw_model.get_reward(data, total_generated_texts, num_return_sequences, dataset)

if args.inference:
    for s in data:
        s["output"] = s["output"][0]
        s["gpt_output"] = s["gpt_output"][0]
    if dataset == "combined":
        for name in ["asqa", "qampari", "eli5"]:
            samples = [s for s in data if s["dataset"] == name]
            with open(save_path.replace("combined_result.json", "{}_result.json".format(name)), "w") as samples_json:
                json.dump({"data": samples}, samples_json, indent=4)
    else:
        with open(save_path, "w") as tgt_json:
            json.dump({"data": data}, tgt_json, indent=4)
else:
    with open(save_path, "w") as tgt_json:
        json.dump(data, tgt_json, indent=4)
