# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the policy functions for the fine-grained RL
#  Referenced: https://github.com/liujch1998/rainier/
#  All Rights Reserved.
#  *********************************************************

from typing import Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, BitsAndBytesConfig, AutoModelForCausalLM
from typing import Optional, List, Iterable, Dict, Any, Tuple
from .utils import logits_to_entropy, mask_pad, load_model, remove_prompt_and_pad
from .reward_utils import remove_citations
import copy
import os

from peft import PeftModel, prepare_model_for_kbit_training

from fgrlhf.utils import get_qlora_config

class T5Policy:

    def __init__(self,
                 model_ckpt: str,
                 tokenizer,
                 policy_value_sharing: bool,
                 accelerator,
                 reference=False,
                 rlhf_ckpt=None
                ):
        self.tokenizer = tokenizer
        self.policy_value_sharing = policy_value_sharing
        self.accelerator = accelerator

        if reference:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt, quantization_config=bnb_config)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        
        # regression head for policy-value sharing
        self.linear = torch.nn.Linear(self.model.config.d_model, 1)    

        if rlhf_ckpt is not None and not reference:
            self.model.load_state_dict(torch.load(rlhf_ckpt)["model"])
            self.linear.load_state_dict(torch.load(rlhf_ckpt)["linear"])

        self.model.eval()

        self.synced_gpus = self.accelerator.num_processes > 1
        
    def sample(self,
               prompts_input_ids: torch.Tensor, # (B, input_len)
               prompts_attention_mask: torch.Tensor, # (B, input_len)
               do_sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               num_beams: int = 1,
               num_return_sequences: int = 1,
              ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        
        prompts_text = self.tokenizer.batch_decode(prompts_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        if do_sample:
            generated_input_ids = unwrapped_model.generate(
                input_ids=prompts_input_ids,
                attention_mask=prompts_attention_mask,
                max_length=self.tokenizer.max_generated_len + 1,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                synced_gpus=self.synced_gpus,
            ) # begins with 0 ([BOS]); ends with 1 ([EOS])
            
        else:
            generated_input_ids = unwrapped_model.generate(
                input_ids=prompts_input_ids,
                attention_mask=prompts_attention_mask,
                max_length=self.tokenizer.max_generated_len + 1,
                num_beams=num_beams,
                do_sample=False,
                num_return_sequences=num_return_sequences,
                synced_gpus=self.synced_gpus,
            )

        generated_input_ids = generated_input_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])

        generated_input_ids = F.pad(generated_input_ids, (0, self.tokenizer.max_generated_len - generated_input_ids.size(1)), value=self.tokenizer.pad_token_id) # (B, output_len)
        generated_attention_mask = (generated_input_ids != self.tokenizer.pad_token_id).long()
        generated_text = self.tokenizer.batch_decode(generated_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


        # repeat input sequences for num_return_sequences times
        prompts_text = [elem for elem in prompts_text for _ in range(num_return_sequences)]
        
        # Preprocessing adapted from ALCE. Retain only the first line. Remove end token.
        generated_texts_for_reward = copy.deepcopy(generated_text)
        for i in range(len(generated_texts_for_reward)):
            generated_texts_for_reward[i] = generated_texts_for_reward[i].strip().split("\n")[0] # If too many answers have new lines, then maybe this line needs to be removed. Currently, correctness sequence level reward uses normalized text
            generated_texts_for_reward[i] = generated_texts_for_reward[i].replace("<|im_end|>", "")
        
        # Remove all citations for all non-AutoAIS evaluation
        normalized_generated_text = generated_texts_for_reward
        for i in range(len(normalized_generated_text)):
            normalized_generated_text[i] = remove_citations(normalized_generated_text[i])

        return {
            'prompts_text': prompts_text,
            'prompts_input_ids': prompts_input_ids.repeat_interleave(num_return_sequences, dim=0), # (B, input_len)
            'prompts_attention_mask': prompts_attention_mask.repeat_interleave(num_return_sequences, dim=0), # (B, input_len)
            'generated_text': generated_text,
            'generated_input_ids': generated_input_ids, # (B, output_len)
            'generated_attention_mask': generated_attention_mask, # (B, output_len)
            # 'generated_texts_for_reward': generated_texts_for_reward,
            'normalized_generated_text': normalized_generated_text,
        }
    

    def forward_pass(self,
                     prompts_input_ids: torch.Tensor, # (B, input_len)
                     prompts_attention_mask: torch.Tensor, # (B, input_len)
                     generated_input_ids: torch.Tensor, # (B, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len)
                    ):

        outputs = self.model(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        generated_logits = outputs.logits # (B, output_len, V)
        logprobs = F.log_softmax(generated_logits, dim=-1)
        generated_logprobs = torch.gather(logprobs, 2, generated_input_ids[:, :, None]).squeeze(2) # (B, output_len)
        generated_entropy = logits_to_entropy(generated_logits) # (B, output_len)

        results = {
            'generated_logits': generated_logits, # (B, output_len, V)
            'generated_logprobs': mask_pad(generated_logprobs, generated_attention_mask), # (B, output_len)
            'generated_entropy': mask_pad(generated_entropy, generated_attention_mask), # (B, output_len)
        }

        if self.policy_value_sharing:
            logits = self.linear(outputs.decoder_hidden_states[-1]).squeeze(-1) # (B, output_len)
            results.update({
                'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, output_len)
            })

        return results



class LLAMAPolicy:

    def __init__(self,
                 model_ckpt: str,
                 peft_ckpt: str,
                 tokenizer,
                 policy_value_sharing: bool,
                 accelerator,
                 reference=False,
                 rlhf_ckpt=None
                ):
        self.tokenizer = tokenizer
        self.policy_value_sharing = policy_value_sharing
        self.accelerator = accelerator

        model = load_model(model_ckpt, tokenizer, init_sft=False)
        if not reference:
            model = prepare_model_for_kbit_training(model)

        if rlhf_ckpt is not None and not reference:
            peft_ckpt = os.path.join(os.path.dirname(rlhf_ckpt), "best_policy_peft")
        self.model = PeftModel.from_pretrained(model, peft_ckpt, is_trainable=not reference)
        self.model.print_trainable_parameters()

        # regression head for policy-value sharing
        self.linear = torch.nn.Linear(self.model.config.d_model, 1)    

        if rlhf_ckpt is not None and not reference:
            self.linear.load_state_dict(torch.load(rlhf_ckpt)["linear"])


        self.model.eval()

        self.synced_gpus = self.accelerator.num_processes > 1
        
    def sample(self,
               prompts_input_ids: torch.Tensor, # (B, input_len)
               prompts_attention_mask: torch.Tensor, # (B, input_len)
               do_sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               num_beams: int = 1,
               num_return_sequences: int = 1,
              ) -> Dict[str, Union[torch.Tensor, List[str], List[int]]]:
        
        prompts_text = self.tokenizer.batch_decode(prompts_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        prompts_len = torch.sum(prompts_attention_mask, dim=1).tolist()

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        # unwrapped_model = self.model
        
        if do_sample:
            combined_input_ids = unwrapped_model.generate(
                input_ids=prompts_input_ids,
                attention_mask=prompts_attention_mask,
                max_new_tokens=self.tokenizer.max_generated_len + 1, # Replace max_length with max_new_tokens
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                synced_gpus=self.synced_gpus,
            ) # begins with 0 ([BOS]); ends with 1 ([EOS])
            
        else:
            combined_input_ids = unwrapped_model.generate(
                input_ids=prompts_input_ids,
                attention_mask=prompts_attention_mask,
                max_new_tokens=self.tokenizer.max_generated_len + 1, # Replace max_length with max_new_tokens
                num_beams=num_beams,
                do_sample=False,
                num_return_sequences=num_return_sequences,
                synced_gpus=self.synced_gpus,
            )

        # begins with 0 ([BOS]); ends with 1 ([EOS])
        combined_attention_mask = (combined_input_ids != self.tokenizer.pad_token_id).long()
        
        # generated_input_ids should start with a token in the format _string, and the space should not be a separate token, even if such space is manually added in init sft
        generated_input_ids = remove_prompt_and_pad(combined_input_ids, prompts_len, self.tokenizer.max_generated_len, self.tokenizer.pad_token_id)

        generated_attention_mask = (generated_input_ids != self.tokenizer.pad_token_id).long()
        generated_text = self.tokenizer.batch_decode(generated_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


        # repeat input sequences for num_return_sequences times
        prompts_text = [elem for elem in prompts_text for _ in range(num_return_sequences)]
        
        # Preprocessing adapted from ALCE. Retain only the first line. Remove end token.
        generated_texts_for_reward = copy.deepcopy(generated_text)
        for i in range(len(generated_texts_for_reward)):
            generated_texts_for_reward[i] = generated_texts_for_reward[i].strip().split("\n")[0] # If too many answers have new lines, then maybe this line needs to be removed. Currently, correctness sequence level reward uses normalized text
            generated_texts_for_reward[i] = generated_texts_for_reward[i].replace("<|im_end|>", "")
        
        # Remove all citations for all non-AutoAIS evaluation
        normalized_generated_text = generated_texts_for_reward
        for i in range(len(normalized_generated_text)):
            normalized_generated_text[i] = remove_citations(normalized_generated_text[i])

        return {
            'prompts_text': prompts_text,
            'prompts_input_ids': prompts_input_ids.repeat_interleave(num_return_sequences, dim=0), # (B, input_len)
            'prompts_attention_mask': prompts_attention_mask.repeat_interleave(num_return_sequences, dim=0), # (B, input_len)
            'combined_input_ids': combined_input_ids,
            'combined_attention_mask': combined_attention_mask,
            'generated_text': generated_text,
            'generated_input_ids': generated_input_ids, # (B, output_len)
            'generated_attention_mask': generated_attention_mask, # (B, output_len)
            # 'generated_texts_for_reward': generated_texts_for_reward,
            'normalized_generated_text': normalized_generated_text,
            'prompts_len': prompts_len,
        }
    

    def forward_pass(self,
                     prompts_len: List[int],
                     combined_input_ids: torch.Tensor, # (B, combined_len), includes [BOS]
                     combined_attention_mask: torch.Tensor, # (B, combined_len)
                     generated_input_ids: torch.Tensor, # (B, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len)
                    ):

        outputs = self.model(
            input_ids=combined_input_ids, # TODO: Check whether combined_input_ids needs to start with [BOS] or not
            attention_mask=combined_attention_mask,
            # labels=mask_pad(generated_input_ids, generated_attention_mask, -100), # The labels for llama are optional and used for mlm only
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        combined_logits = outputs.logits # (B, combined_len, V)
        generated_logits = remove_prompt_and_pad(combined_logits, prompts_len, self.tokenizer.max_generated_len, self.tokenizer.pad_token_id) # (B, output_len, V)
        logprobs = F.log_softmax(generated_logits, dim=-1)
        generated_logprobs = torch.gather(logprobs, 2, generated_input_ids[:, :, None]).squeeze(2) # (B, output_len)
        generated_entropy = logits_to_entropy(generated_logits) # (B, output_len)

        results = {
            'generated_logits': generated_logits, # (B, output_len, V)
            'generated_logprobs': mask_pad(generated_logprobs, generated_attention_mask), # (B, output_len)
            'generated_entropy': mask_pad(generated_entropy, generated_attention_mask), # (B, output_len)
        }

        if self.policy_value_sharing:
            hidden_states = outputs.hidden_states[-1]
            hidden_states = remove_prompt_and_pad(hidden_states, prompts_len, self.tokenizer.max_generated_len, self.tokenizer.pad_token_id) # (B, output_len, d)
            logits = self.linear(hidden_states).squeeze(-1) # (B, output_len)
            results.update({
                'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, output_len)
            })

        return results
