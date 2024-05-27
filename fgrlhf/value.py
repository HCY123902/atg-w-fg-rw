# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the value functions for the fine-grained RL
#  Referenced: https://github.com/liujch1998/rainier/
#  All Rights Reserved.
#  *********************************************************


from typing import Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from typing import Optional, List, Iterable, Dict, Any, Tuple
from .utils import logits_to_entropy, mask_pad, load_model, remove_prompt_and_pad

import os

from peft import PeftModel, prepare_model_for_kbit_training

class MLP(torch.nn.Module):
    
    def __init__(self, d_model, d_out) -> None:
        super().__init__()
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, d_out),
        )
    
    def forward(self, x):
        return self.model(x)


class T5Value:

    def __init__(self,
                 model_ckpt: str,
                 model,
                 tokenizer,
                 accelerator,
                 freeze_model: bool = False,
                 rlhf_ckpt=None,
                ):
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        
        if model is not None:
            self.model = model
            return

        self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
            
        self.linear = MLP(self.model.config.d_model, 1)

        if rlhf_ckpt is not None:
            self.model.load_state_dict(torch.load(rlhf_ckpt)["value_model"])
            self.linear.load_state_dict(torch.load(rlhf_ckpt)["value_linear"])
        
        # freeze all parameters except the last layer
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False


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

        logits = self.linear(outputs.decoder_hidden_states[-1]).squeeze(-1) # (B, output_len)
        results = {
                'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, output_len)
        }

        return results



class LLAMAValue:

    def __init__(self,
                 model_ckpt: str,
                 peft_ckpt: str,
                 model,
                 tokenizer,
                 accelerator,
                 freeze_model: bool = False,
                 rlhf_ckpt=None
                ):
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        
        if model is not None:
            self.model = model
            return

        model = load_model(model_ckpt, tokenizer, init_sft=False)

        model = prepare_model_for_kbit_training(model)

        if rlhf_ckpt is not None:
            peft_ckpt = os.path.join(os.path.dirname(rlhf_ckpt), "best_value_peft")
        self.model = PeftModel.from_pretrained(model, peft_ckpt, is_trainable=True)
        self.model.print_trainable_parameters()

        self.linear = MLP(self.model.config.d_model, 1)   

        if rlhf_ckpt is not None:
            self.linear.load_state_dict(torch.load(rlhf_ckpt)["value_linear"])
        
        # freeze all parameters except the last layer
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward_pass(self,
                     prompts_len: List[int],
                     combined_input_ids: torch.Tensor, # (B, combined_len), includes [BOS]
                     combined_attention_mask: torch.Tensor, # (B, combined_len)
                     generated_input_ids: torch.Tensor, # (B, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len)
                    ):

        outputs = self.model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            # labels=mask_pad(generated_input_ids, generated_attention_mask, -100), # The labels for llama are optional and used for mlm only
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1] # (B, combined_len, d)
        hidden_states = remove_prompt_and_pad(hidden_states, prompts_len, self.tokenizer.max_generated_len, self.tokenizer.pad_token_id) # (B, output_len, d)

        logits = self.linear(hidden_states).squeeze(-1) # (B, output_len)
        results = {
                'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, output_len)
        }

        return results