import argparse
from collections import defaultdict
from itertools import chain
import json
import logging
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
import accelerate
import wandb

import yaml
import nltk

from fgrlhf.ppo import PPOTrainer
from fgrlhf.policy import T5Policy, LLAMAPolicy
from fgrlhf.value import T5Value, LLAMAValue
from fgrlhf.utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp


from reward import FineGrainedReward, FineGrainedRewardCitation, NLIModel

import sys

from peft import PeftConfig

os.environ["WANDB__SERVICE_WAIT"] = "300"

log_file_path = os.environ["TRAIN_LOG_PATH"]

log_fh = open(log_file_path, 'a')

sys.stderr = log_fh
sys.stdout = log_fh

logging.basicConfig(level=logging.ERROR, filename=log_file_path, filemode="a")

# prepare accelerator and logger
accelerator = accelerate.Accelerator()
device = accelerator.device
log = accelerate.logging.get_logger(__name__, log_level='INFO')
def log_info(s, main_only=True):
    if accelerator.is_main_process:
        log.info(s)
        
# load parameters
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="path to config file")
args = parser.parse_args()
# load yaml file
with open(args.config) as f:
    args =yaml.safe_load(f)


# prepare data
class TextGenDataset(Dataset):
    def __init__(self, split, tokenizer, accelerator=None, length_limit=None, dataset_name=None, model_type="t5"):
        super().__init__()
        
        self.split = split


        self.dataset_map = {
            "asqa": {
                "train": "tasks/qa_feedback/data/asqa_train.json",
                "dev": "tasks/qa_feedback/data/asqa_dev.json",
                "test": "tasks/qa_feedback/data/asqa_test.json"
            },
            "qampari": {
                "train": "tasks/qa_feedback/data/qampari_train.json",
                "dev": "tasks/qa_feedback/data/qampari_dev.json",
                "test": "tasks/qa_feedback/data/qampari_test.json"
            },
            "eli5": {
                "train": "tasks/qa_feedback/data/eli5_train.json",
                "dev": "tasks/qa_feedback/data/eli5_dev.json",
                "test": "tasks/qa_feedback/data/eli5_test.json"
            },
            "combined": {
                "train": "tasks/qa_feedback/data/combined_train.json",
                "dev": "tasks/qa_feedback/data/combined_dev.json",
                "test": "tasks/qa_feedback/data/combined_test.json"
            }
        }

        self.dataset_fns = self.dataset_map[dataset_name]

        log_info(f'Currently loading split {self.split} from {dataset_name}')
        
        self.n_card = 1
        if accelerator is not None:
            self.n_card = accelerator.num_processes
        
        
        self.tokenizer = tokenizer

        self.instances = self.load_datasets()
        
        if length_limit is not None:
            self.instances = self.instances[:length_limit]

        if split == 'train':
            random.shuffle(self.instances)

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
        
        log_info(f'Loaded split {self.split} with {len(instances)} total instances')
        
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
    

def main():
    # set seed
    set_seed(args['train']['seed'], args['train']['cuda_deterministic'])
    
    # set saving directories
    log_info(f"Write to output directory: {args['logging']['save_dir']}")
    
    if accelerator.is_main_process:
        ensure_dir(args['logging']['save_dir'])
        # save the config file
        with open(os.path.join(args['logging']['save_dir'], 'args.json'), 'w') as f:
            json.dump(args, f, indent=2)

    model_type = (args['model']['type'] if 'type' in args['model'] else "t5")

    if model_type != "t5":
        raise Exception("Model {} is not implemented".format(model_type))

    if 'peft_ckpt' in args['model']['policy_model']:
        peft_config = PeftConfig.from_pretrained(args['model']['policy_model']['peft_ckpt'])
    
    # initialize policy and value model tokenizers
    if model_type == "llama":        
        raise Exception("Model {} is not implemented".format(model_type))

    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args['model']['policy_model']['ckpt'], 
                                                            model_max_length=args['env']['max_input_len'])
    
    tokenizer.padding_side = args['model']['policy_model']['input_padding_side']
    tokenizer.max_input_len = args['env']['max_input_len']
    tokenizer.max_generated_len = args['env']['max_generated_len']
        
    
    dataset_name = args['dataset']['name']
    
    args['model']['type'] = model_type

    # Load data
    log_info(f'Loading data ...')
    train_dataset = TextGenDataset( 'train', tokenizer, accelerator=accelerator, dataset_name=dataset_name, model_type=model_type)
    # train ds is shuffled in its constructor
    train_dataloader = DataLoader(train_dataset, batch_size=args['train']['sampling_batch_size_per_card'], 
                                  shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)

    eval_dataset = TextGenDataset( 'dev',  tokenizer, accelerator=accelerator, length_limit=None, dataset_name=dataset_name, model_type=model_type)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args['train']['sampling_batch_size_per_card'], 
                                 shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)

    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)


    # Initialize models and optimizer
    log_info(f'Initializing models ...')

    if model_type == "t5":
        ref_policy = T5Policy(
            model_ckpt=args['model']['policy_model']['ckpt'],
            tokenizer=tokenizer,
            policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
            accelerator=accelerator,
            reference=True
        )
    elif model_type == "llama":
        raise Exception("Model {} is not implemented".format(model_type))

    else:
        raise Exception("Model type {} is not valid".format(model_type))

    ref_policy.model, ref_policy.linear = accelerator.prepare(ref_policy.model, ref_policy.linear)

    rlhf_ckpt = args['model']['rlhf_ckpt'] if 'rlhf_ckpt' in args['model'] else None
    if rlhf_ckpt is not None:
        log_info("Continuing training with rlhf_ckpt {}".format(rlhf_ckpt))

    if model_type == "t5":
        policy = T5Policy(
            model_ckpt=args['model']['policy_model']['ckpt'],
            tokenizer=tokenizer,
            policy_value_sharing=args['model']['value_model']['policy_value_sharing'],
            accelerator=accelerator,
            rlhf_ckpt=rlhf_ckpt,
        )
    elif model_type == "llama":
        raise Exception("Model {} is not implemented".format(model_type))

    else:
        raise Exception("Model type {} is not valid".format(model_type))

    policy.model, policy.linear = accelerator.prepare(policy.model, policy.linear)
    
    if model_type == "t5":
        value = T5Value(
            model_ckpt=args['model']['value_model']['ckpt'],
            model=policy.model if args['model']['value_model']['policy_value_sharing'] else None,
            tokenizer=tokenizer,
            accelerator=accelerator,
            freeze_model=False if args['model']['value_model']['policy_value_sharing'] else args['model']['value_model']['freeze_value_model'],
            rlhf_ckpt=rlhf_ckpt,
        )
    elif model_type == "llama":
        raise Exception("Model {} is not implemented".format(model_type))

    else:
        raise Exception("Model type {} is not valid".format(model_type))

    if not args['model']['value_model']['policy_value_sharing']:
        value.model, value.linear = accelerator.prepare(value.model, value.linear)

    nli_model = NLIModel(args['reward']['nli_model']['ckpt'], log_info=log_info, accelerator=accelerator)
    
    reward = FineGrainedRewardCitation(
        tokenizer=tokenizer,
        kl_coef=args['ppo']['kl_coef'],
        citation_recall_positive_reward = args['reward']['citation_model']['recall_positive_reward'],
        citation_recall_negative_reward = args['reward']['citation_model']['recall_negative_reward'],
        citation_precision_positive_reward = args['reward']['citation_model']['precision_positive_reward'],
        citation_precision_negative_reward = args['reward']['citation_model']['precision_negative_reward'],
        at_most_citations = args['reward']['citation_model']['at_most_citations'],
        correctness_positive_reward = args['reward']['correctness_model']['positive_reward'],
        correctness_negative_reward = args['reward']['correctness_model']['negative_reward'],
        fluency_reward_mean = args['reward']['fluency_model']['mean'],
        fluency_reward_std = args['reward']['fluency_model']['std'],
        fluency_reward_bias = args['reward']['fluency_model']['bias'],
        fluency_reward_scale = args['reward']['fluency_model']['scale'],
        # sep = "</s>"
        nli_model=nli_model,
        holistic=("holistic" in args['reward'] and args['reward']["holistic"]),
        citation_recall=("exclude_citation_recall" not in args['reward']['citation_model'] or not args['reward']['citation_model']["exclude_citation_recall"]),
        citation_precision=("exclude_citation_precision" not in args['reward']['citation_model'] or not args['reward']['citation_model']["exclude_citation_precision"]),
        correctness_recall=("exclude_correctness_recall" not in args['reward']['correctness_model'] or not args['reward']['correctness_model']["exclude_correctness_recall"])
    )

    nli_model.model = accelerator.prepare(nli_model.model)
    
    # prepare optimizers and schedulers
    if args['model']['value_model']['policy_value_sharing']:
        parameters = chain(policy.model.parameters(), policy.linear.parameters())
    else:
        parameters = chain(policy.model.parameters(), policy.linear.parameters(), value.model.parameters(), value.linear.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args['train']['lr'], eps=1e-5)

    # Number of batches in each process
    total_steps = ceil_div(args['train']['total_episodes'], 
                                args['train']['sampling_batch_size_per_card'] * accelerator.num_processes * args['env']['train_num_samples_per_input'])
    
    scheduler = transformers.get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=100*args['train']['n_ppo_epoch_per_rollout']*accelerator.num_processes, # The first 100 batches in each process are used for warming up
        num_training_steps=total_steps*args['train']['n_ppo_epoch_per_rollout']*accelerator.num_processes,
    )
    
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)


    # Set up trainer
    trainer = PPOTrainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        ref_policy_model=ref_policy,
        policy_model=policy,
        value_model=value,
        reward_model=reward,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        log_info=log_info,
        model_type=model_type,
    )
    
    # Each step corresponds to 1 batch in each process
    steps = list(range(total_steps + 1))
    steps = tqdm(steps) if accelerator.is_main_process else steps
    for step in steps:
        trainer.train(step)

        # Synchronize after training with each batch
        accelerator.wait_for_everyone()
        # early stopping because KL explodes
        if trainer.should_early_stop:
            if accelerator.is_local_main_process:
                print("Early stopping triggered. Terminating training.")
            break
            
if __name__ == '__main__':
    main()
