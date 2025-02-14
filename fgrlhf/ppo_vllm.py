import argparse
from collections import defaultdict
from itertools import chain
import json
import logging
from select import select
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
from typing import Dict
import yaml
import nltk
from typing import Optional, List, Iterable, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
import accelerate
import wandb

from .utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp

from fgrlhf.reward_utils import normalize_answer, remove_citations

from peft import get_peft_model_state_dict

log_file_path = os.environ["TRAIN_LOG_PATH"]

logging.basicConfig(level=logging.ERROR, filename=log_file_path, filemode="a")

class PPOTrainerVLLM:
    def __init__(self,
                 args: argparse.Namespace,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 model,
                #  policy_model,
                #  value_model,
                 reward_model,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                #  accelerator: accelerate.Accelerator,
                 log_info,
                 model_type,
                ):
        
        self.log_info = log_info
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # early stopping if KL too big
        self.should_early_stop = False
        self.huge_kl_count = 0

        self.batchify = lambda x, n: [x[i:i + n] for i in range(0, len(x), n)]
        
        # if self.accelerator.is_main_process:
        if args['logging']['wandb_log']:
            wandb.init(entity=args["logging"]["wandb_entity"], project=args["logging"]["wandb_project"], name=args['logging']['run_name'], config=args, settings=wandb.Settings(start_method="fork"))
        else:
            wandb.init(config=args, mode='disabled', settings=wandb.Settings(start_method="fork"))
        
        wandb.define_metric('train/step')
        wandb.define_metric('eval/step')
        wandb.define_metric('train/*', step_metric='train/step')
        wandb.define_metric('eval/*', step_metric='eval/step', summary='max')

        self.train_sampler = iter(self.train_dataloader)
        for _ in range(len(self.train_dataloader)):
            next(self.train_sampler)

        self.eval_accs = {}

        if "save_corr_rec" in self.args["logging"] and self.args["logging"]["save_corr_rec"]:
            self.eval_corr_rec = {}

        self.model_type=model_type
        

    def compute_advantages(self, results, num_samples):
        
        old_values = results['generated_value']
        rewards = results['rewards/penalized']
        mask = results['generated_attention_mask'] # (B, KL)
        
        with torch.no_grad():
            if self.args['ppo']['whiten_rewards']:
                whitened_rewards = whiten(rewards, mask, shift_mean=False, accelerator=None)
            else:
                whitened_rewards = rewards
            
            lastgaelam = 0
            advantages_reversed = []

            # The longest generation length in the batch
            gen_length = mask.sum(dim=1).max().item()
            for t in reversed(range(gen_length)):
                nextvalues = old_values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = whitened_rewards[:, t] + self.args['ppo']['gamma'] * nextvalues - old_values[:, t]
                lastgaelam = delta + self.args['ppo']['gamma'] * self.args['ppo']['lam'] * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            advantages = F.pad(advantages, (0, whitened_rewards.size(1) - gen_length), value=0.0)
            
            
            # After the rough cancellation between the previous self.args['ppo']['gamma'] * nextvalues and the next old_values[:, t], this equals V^{targ}, which is sum_{t'=t}^{T-1} gamma^{t'-t}r_t + Vold^{T-t}
            returns = advantages + old_values
            
            whitened_advantages = advantages.detach()
            whitened_advantages = whiten(advantages, mask, accelerator=None).detach()

            
        results['whitened_advantages'] = whitened_advantages
        results['returns'] = returns
                

    def loss(self, results, all_mask_weight):
        
        old_values = results['generated_value']
        old_logprobs = results['generated_logprobs']
        mask = results['generated_attention_mask'] # (B, KL)
        
        whitened_advantages = results['whitened_advantages']
        returns = results['returns']

        weight = mask.sum(dim=1).float().mean().item() / all_mask_weight

        forward_inputs = {
            'prompts_len': results['prompts_len'],
            'combined_input_ids': results['combined_input_ids'],
            'combined_attention_mask': results['combined_attention_mask'],
            'generated_input_ids': results['generated_input_ids'],
            'generated_attention_mask': results['generated_attention_mask'],
        }

        policy_forward = self.model.forward_pass(**forward_inputs)
        new_logprobs = policy_forward['generated_logprobs']

        # new_logprobs: Probabilities at current ppo epoch in the batch; old_logprobs: Probabilites at the previous ppo epoch in the batch; ref_logprobs: Probabilities provided by the initial model after supervised training
        # The ratio here is measured in terns of the distance between the model at the current ppo epoch and the model at the previous ppo epoch
        ratio = torch.exp(new_logprobs - old_logprobs)
        pg_losses1 = -whitened_advantages * ratio
        pg_losses2 = -whitened_advantages * torch.clamp(ratio, min=1.0 - self.args['ppo']['cliprange'], max=1.0 + self.args['ppo']['cliprange'])
        pg_loss = reduce_mean(torch.max(pg_losses1, pg_losses2), mask)
        pg_loss = pg_loss * weight

        discounted_pg_loss = self.args['ppo']['pg_coef'] * pg_loss

        discounted_pg_loss.backward()

        if self.args['model']['value_model']['policy_value_sharing']:
            new_values = policy_forward['generated_value']
        else:
            value_forward = self.model.value_forward_pass(**forward_inputs)
            new_values = value_forward['generated_value']
            new_values *= mask

        new_values_clipped = clamp(new_values, old_values - self.args['ppo']['cliprange_value'], old_values + self.args['ppo']['cliprange_value'])
        vf_losses1 = torch.square(new_values - returns)
        vf_losses2 = torch.square(new_values_clipped - returns)
        vf_loss = .5 * reduce_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_loss = vf_loss * weight

        discounted_vf_loss = self.args['ppo']['vf_coef'] * vf_loss

        discounted_vf_loss.backward()

        loss = self.args['ppo']['pg_coef'] * pg_loss + self.args['ppo']['vf_coef'] * vf_loss

        results['loss/total'] = loss
        results['loss/policy'] = pg_loss
        results['loss/value'] = vf_loss

    def train(self, step):
        # Save the last checkpoint and the best checkpoint. If they happen to be the same, remove the duplicate last.pth accordingly
        if step % self.args['train']['eval_interval'] == 0:
            self.save(step=step)
            self.valid(step=step)

        try:
            batch = next(self.train_sampler)
        except StopIteration:
            self.train_sampler = iter(self.train_dataloader)
            batch = next(self.train_sampler)
        
        self.model.model.eval()
        self.model.linear.eval()
        
        # Rollout from current policy
        with torch.no_grad():
            results = self.model.sample(
                prompts_input_ids=batch['prompts_input_ids'],
                prompts_attention_mask=batch['prompts_attention_mask'],
                num_return_sequences=self.args['env']['train_num_samples_per_input'],
                **self.args['model']['policy_model']['train_generation_kwargs'],
            )
    
        forward_inputs = {
            'prompts_len': results['prompts_len'],
            'combined_input_ids': results['combined_input_ids'],
            'combined_attention_mask': results['combined_attention_mask'],
            'generated_input_ids': results['generated_input_ids'],
            'generated_attention_mask': results['generated_attention_mask'],
        }

        
        with torch.no_grad():
            policy_forward = self.model.forward_pass(**forward_inputs)
            results.update(policy_forward)
        
        # Run value network
        if not self.args['model']['value_model']['policy_value_sharing']:
            with torch.no_grad(): # treat the values at beginning of step as ground-truth
                value_forward = self.model.value_forward_pass(**forward_inputs)
                results['generated_value'] = value_forward['generated_value']
                results['generated_value'] *= results['generated_attention_mask']  # TODO: I doubt if this line is necessary

        # Run ref policy
        with torch.no_grad():
            ref_policy_forward = self.model.ref_p_forward_pass(**forward_inputs)
            results['generated_ref_logits'] = ref_policy_forward['generated_logits']
            results['generated_ref_logprobs'] = ref_policy_forward['generated_logprobs']
        
        # Get reward
        with torch.no_grad():
            reward_results = self.reward_model.get_reward(
                prompts_input_ids=results['prompts_input_ids'],
                prompts_attention_mask=results['prompts_attention_mask'],
                generated_input_ids=results['generated_input_ids'],
                generated_attention_mask=results['generated_attention_mask'],
                generated_texts=results['generated_text'],
                # generated_texts=results['generated_texts_for_reward'],
                # generated_texts_for_reward=results['generated_texts_for_reward'],
                normalized_generated_text=results['normalized_generated_text'],
                metadata = [elem for elem in batch['metadata'] for _ in range(self.args['env']['train_num_samples_per_input'])],
                print_reward = (step % self.args['train']['eval_interval']) < 1 and (self.reward_model.ablation or self.reward_model.holistic)
            )
            results.update(reward_results)
            self.reward_model.kl_penalize_reward(results)

        # Get advantages
        self.compute_advantages(results, self.args['env']['train_num_samples_per_input'])
        
        n_results = len(results['generated_input_ids'])
        
        loss_totals, loss_policies, loss_values =  [], [], []
        reward_penalizeds, reward_kls, reward_raws = [], [], []
        
        # Train
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        
        self.model.model.train()
        self.model.linear.train()
        
        for ppo_epoch_idx in range(self.args['train']['n_ppo_epoch_per_rollout']):
            self.optimizer.zero_grad()
            
            # get the weight for each sub-batch
            mask = results['generated_attention_mask']
            all_mask = mask
            all_mask_weight = all_mask.sum(dim=1).float().mean().item()
            
            for batch_idx in range(0,n_results, self.args['train']['training_batch_size_per_card']):
                batch_results = {}
                
                for k, v in results.items():
                    batch_results[k] = v[batch_idx:batch_idx+self.args['train']['training_batch_size_per_card']]
            

                self.loss(batch_results, all_mask_weight)
                

                # logging
                if ppo_epoch_idx == self.args['train']['n_ppo_epoch_per_rollout'] - 1:
                    loss_total = batch_results['loss/total'].unsqueeze(0) # (1)
                    loss_policy = batch_results['loss/policy'].unsqueeze(0) # (1)
                    loss_value = batch_results['loss/value'].unsqueeze(0) # (1)
                    reward_penalized = torch.mean(reduce_sum(batch_results['rewards/penalized'], batch_results['generated_attention_mask'], axis=1)).unsqueeze(0) # (1)
                    reward_kl = torch.mean(reduce_sum(batch_results['rewards/kl'], batch_results['generated_attention_mask'], axis=1)).unsqueeze(0) # (1)
                    reward_raw =  torch.mean(reduce_sum(batch_results['rewards/raw'], batch_results['generated_attention_mask'], axis=1)).unsqueeze(0) # (1)

                    loss_totals.append(loss_total)
                    loss_policies.append(loss_policy)
                    loss_values.append(loss_value)
                    reward_penalizeds.append(reward_penalized)
                    reward_kls.append(reward_kl)
                    reward_raws.append(reward_raw)
                    
                
            if self.args['train']['clip_grad']:
                torch.nn.utils.clip_grad_norm_(
                    chain(
                        [p for _, p in self.model.model.named_parameters() if p.requires_grad],
                        self.model.linear.parameters()
                    ),
                    self.args['train']['max_grad_norm'])
                
            self.optimizer.step()
            self.scheduler.step()

        loss_total = torch.cat(loss_totals, dim=0)
        loss_policy = torch.cat(loss_policies, dim=0)
        loss_value = torch.cat(loss_values, dim=0)
        reward_penalized = torch.cat(reward_penalizeds, dim=0)
        reward_kl = torch.cat(reward_kls, dim=0)
        reward_raw = torch.cat(reward_raws, dim=0)


        losses_total = loss_total # (num_gpus)
        losses_policy = loss_policy # (num_gpus)

        losses_value = loss_value # (num_gpus)
        rewards_penalized = reward_penalized # (num_gpus)
        rewards_kl = reward_kl # (num_gpus)
        rewards_raw = reward_raw # (num_gpus)

        loss_total = losses_total.mean().item()
        loss_policy = losses_policy.mean().item()

        loss_value = losses_value.mean().item()
        reward_penalized = rewards_penalized.mean().item()
        reward_kl = rewards_kl.mean().item()
        reward_raw = rewards_raw.mean().item()
    
        # Logging
        if self.args['logging']['wandb_log']:

            this_batch_kl = np.mean(reward_kl)

            if step % self.args['logging']['log_interval'] == 0:
                wandb.log({
                    'train/step': step,
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/loss/total': np.mean(loss_total),
                    'train/loss/policy': np.mean(loss_policy),
                    'train/loss/value': np.mean(loss_value),
                    'train/reward/penalized': np.mean(reward_penalized),
                    'train/reward/KL': this_batch_kl,
                    'train/reward/raw': np.mean(reward_raw),
                })
                
            if this_batch_kl > self.args['train']['kl_threshold']:
                self.log_info(f"KL divergence {this_batch_kl} exceeds threshold {self.args['train']['kl_threshold']}")
                self.huge_kl_count += 1
                if self.huge_kl_count >= 5:
                    self.should_early_stop = True

        if (step % self.args['train']['eval_interval']) < 1:
            for i in range(self.args['train']['sampling_batch_size_per_card']):
                question = batch['metadata'][i]['question']
                self.log_info("Inspecting sample {} with question {}:".format(i, question))
                value_avg = reduce_mean(results['generated_value'], results['generated_attention_mask'], axis=1)
                rw_raw_sum = reduce_sum(results['rewards/raw'], results['generated_attention_mask'], axis=1)
                rw_kl_sum = reduce_sum(results['rewards/kl'], results['generated_attention_mask'], axis=1)
                rw_kl_pn_sum = reduce_sum(results['rewards/kl_penalty'], results['generated_attention_mask'], axis=1)
                rw_pn_sum = reduce_sum(results['rewards/penalized'], results['generated_attention_mask'], axis=1)
                seq_len = torch.sum(results['generated_attention_mask'], dim=-1)
                for j in range(self.args['env']['train_num_samples_per_input']):
                    curr_idx = i * self.args['env']['train_num_samples_per_input'] + j
                    curr_seq_len = seq_len[curr_idx].item()
                    self.log_info("Sample {} generated text: {}".format(j, results['generated_text'][curr_idx]))
                    self.log_info("Sample {} value: {}".format(j, results['generated_value'][curr_idx, :curr_seq_len]))
                    self.log_info("Sample {} value average: {}".format(j, value_avg[curr_idx].item()))
                    self.log_info("Sample {} reward raw: {}".format(j, results['rewards/raw'][curr_idx, :curr_seq_len]))
                    self.log_info("Sample {} reward raw sum: {}".format(j, rw_raw_sum[curr_idx].item()))
                    self.log_info("Sample {} reward kl: {}".format(j, results['rewards/kl'][curr_idx, :curr_seq_len]))
                    self.log_info("Sample {} reward kl sum: {}".format(j, rw_kl_sum[curr_idx].item()))
                    self.log_info("Sample {} reward kl penality: {}".format(j, results['rewards/kl_penalty'][curr_idx, :curr_seq_len]))
                    self.log_info("Sample {} reward kl penality sum: {}".format(j, rw_kl_pn_sum[curr_idx].item()))
                    self.log_info("Sample {} reward penalized: {}".format(j, results['rewards/penalized'][curr_idx, :curr_seq_len]))
                    self.log_info("Sample {} reward penalized sum: {}".format(j, rw_pn_sum[curr_idx].item()))
                    if self.args['train']['training_batch_size_per_card'] == 1:
                        self.log_info("Sample {} l t: {}".format(j, loss_totals[curr_idx]))
                        self.log_info("Sample {} l p: {}".format(j, loss_policies[curr_idx]))
                        self.log_info("Sample {} l v: {}".format(j, loss_values[curr_idx]))
                    self.log_info("\n\n")


    def valid(self, step):
        self.log_info(f'Evaluating [step {step}] ...')

        self.model.model.eval()
        self.model.linear.eval()
            
        columns=["step", "inputs", "outputs"]
        wandb_table = None
        
        n_entries = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.eval_dataloader)):

                results = self.model.sample(
                    prompts_input_ids=batch['prompts_input_ids'],
                    prompts_attention_mask=batch['prompts_attention_mask'],
                    **self.args['model']['policy_model']['eval_generation_kwargs'],
                )

                forward_inputs = {
                    'prompts_len': results['prompts_len'],
                    'combined_input_ids': results['combined_input_ids'],
                    'combined_attention_mask': results['combined_attention_mask'],
                    'generated_input_ids': results['generated_input_ids'],
                    'generated_attention_mask': results['generated_attention_mask'],
                }

                value_forward = self.model.value_forward_pass(**forward_inputs)
                results['generated_value'] = value_forward['generated_value']


                eval_results = self.reward_model.eval_metrics(
                    prompts_input_ids=results['prompts_input_ids'],
                    prompts_attention_mask=results['prompts_attention_mask'],
                    generated_input_ids=results['generated_input_ids'],
                    generated_attention_mask=results['generated_attention_mask'],
                    generated_texts=results['generated_text'],
                    # generated_texts=results['generated_texts_for_reward'],
                    normalized_generated_text=results['normalized_generated_text'],
                    metadata = batch['metadata'],
                )

                

                results['generated_value_avg'] = reduce_mean(results['generated_value'], results['generated_attention_mask'], axis=1)


                # initialize wandb table if it does not exist
                if wandb_table is None:
                    columns.extend(list(eval_results.keys())) 
                    columns.append("eval/value_start")
                    columns.append("eval/value_avg")
                    wandb_table = wandb.Table(columns=columns)
                
                prompt_inputs = self.model.tokenizer.batch_decode(results['prompts_input_ids'],
                                                                skip_special_tokens=True, 
                                                                clean_up_tokenization_spaces=True)
                
                
                generated_texts = self.model.tokenizer.batch_decode(results['generated_input_ids'],
                                                                skip_special_tokens=True, 
                                                                clean_up_tokenization_spaces=True)

                
                this_data_batch_size = results['prompts_input_ids'].shape[0]
                this_lens = torch.sum(results['generated_attention_mask'], dim=-1)
                
                for batch_i in range(this_data_batch_size):
                    
                    this_entry = [step, prompt_inputs[batch_i], generated_texts[batch_i]]
                    
                    for eval_v in eval_results.values():
                        this_entry.append(eval_v[batch_i])

                    value_start = results['generated_value'][batch_i, 0].item()
                    value_avg = results['generated_value_avg'][batch_i].item()
                    this_entry.append(value_start)
                    this_entry.append(value_avg)
                    
                    wandb_table.add_data(*this_entry)
                    n_entries += 1


    
        # do statistics        
        n_dev_samples = len(wandb_table.data)
        
        stats = {'eval/step': step,
                    f'eval_generation/step_{step}': wandb_table}
        
        value_columns = columns[3:] # the first three are steps, inputs, outputs
        stats.update(self.reward_model.aggregate_metrics(wandb_table, value_columns))
        
        
        if self.args['logging']['wandb_log']:
            wandb.log(stats)

        mean_rewards = stats["eval/rewards"]
    
        self.log_info(f'Evaluated [step {step}] rewards = {mean_rewards:.4f}')
        
        prev_best_step = None if len(self.eval_accs) == 0 else max(self.eval_accs, key=self.eval_accs.get)
        self.eval_accs[step] = mean_rewards
        if prev_best_step is None or mean_rewards > self.eval_accs[prev_best_step]:
            if prev_best_step is not None:
                try:
                    os.remove(f"{self.args['logging']['save_dir']}/ckp_{prev_best_step}.pth")
                except:
                    self.log_info(f'Cannot remove previous best ckpt!')
            shutil.copy(f"{self.args['logging']['save_dir']}/last_ckp_{step}.pth", f"{self.args['logging']['save_dir']}/ckp_{step}.pth")
            self.log_info(f'Best ckpt updated to [step {step}]')
            
            # save best policy again

            if prev_best_step is not None:
                try:
                    shutil.rmtree(f"{self.args['logging']['save_dir']}/best_peft")
                except:
                    self.log_info(f'Cannot remove previous best policy and value peft!')
            self.model.model.save_pretrained(save_directory=f"{self.args['logging']['save_dir']}/best_peft", selected_adapters=[self.model.policy_adapter_name, self.model.value_adapter_name])

        
        if "save_corr_rec" in self.args["logging"] and self.args["logging"]["save_corr_rec"]:
            mean_corr_rec = stats["eval/correctness_recalls"]
            prev_corr_rec_best_step = None if len(self.eval_corr_rec) == 0 else max(self.eval_corr_rec, key=self.eval_corr_rec.get)
            
            self.eval_corr_rec[step] = mean_corr_rec
            if prev_corr_rec_best_step is None or mean_corr_rec > self.eval_corr_rec[prev_corr_rec_best_step]:
                if prev_corr_rec_best_step is not None:
                    try:
                        os.remove(f"{self.args['logging']['save_dir']}/ckp_{prev_corr_rec_best_step}_corr_rec.pth")
                    except:
                        self.log_info(f'Cannot remove previous best ckpt!')
                shutil.copy(f"{self.args['logging']['save_dir']}/last_ckp_{step}.pth", f"{self.args['logging']['save_dir']}/ckp_{step}_corr_rec.pth")
                self.log_info(f'Best corr rec ckpt updated to [step {step}]')

                # save best policy again

                if prev_corr_rec_best_step is not None:
                    try:
                        shutil.rmtree(f"{self.args['logging']['save_dir']}/best_peft_corr_rec")
                    except:
                        self.log_info(f'Cannot remove previous best policy and value peft!')
                self.model.model.save_pretrained(save_directory=f"{self.args['logging']['save_dir']}/best_peft_corr_rec", selected_adapters=[self.model.policy_adapter_name, self.model.value_adapter_name])



    def save(self, step):

        # There is only 1 linear state_dict, used regardless of whether policy_value_sharing is true
        linear_state_dict = self.model.linear.state_dict()

        result = {
            'model': get_peft_model_state_dict(self.model.model, adapter_name=self.model.policy_adapter_name),
            'linear': linear_state_dict,
            'step': step,
            'eval_accs': self.eval_accs,
            # 'optimizer': optimizer_state_dict,
        }
        if not self.args['model']['value_model']['policy_value_sharing']:
            # result['value_linear'] = value_linear_state_dict
            result['value_model'] = get_peft_model_state_dict(self.model.model, adapter_name=self.model.value_adapter_name)

        torch.save(result, f"{self.args['logging']['save_dir']}/last_ckp_{step}.pth")
        self.log_info(f'[step {step}] model checkpoint saved')
