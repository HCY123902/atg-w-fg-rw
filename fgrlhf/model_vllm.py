from typing import Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, BitsAndBytesConfig, AutoModelForCausalLM
from typing import Optional, List, Iterable, Dict, Any, Tuple
from .utils import logits_to_entropy, mask_pad, load_model, remove_prompt_and_pad, VLLM, lora_merge_unmerge_state_dict, concat_prompt_and_generation, DEFAULT_DEVICE_RANK
from .reward_utils import remove_citations
import copy
import os

from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model_state_dict

from fgrlhf.utils import get_qlora_config

from vllm import SamplingParams


from subprocess import check_output

class ModelVLLM:

    def __init__(self,
                 model_ckpt: str,
                 peft_ckpt: str,
                 tokenizer,
                 policy_value_sharing: bool,
                #  accelerator,
                 rlhf_ckpt=None,
                 policy_adapter_name="policy_adapter",
                 value_adapter_name="value_adapter",
                 ref_policy_adapter_name="ref_p_adapter",
                 device_rank=DEFAULT_DEVICE_RANK,
                 log_info=None,
                ):
        self.log_info=log_info
        self.device = torch.device("cuda:{}".format(device_rank))

        self.model_ckpt = model_ckpt
        self.peft_ckpt = peft_ckpt

        self.init_vllm()

        self.stop_tokens = ["\n", "Ċ", "ĊĊ", "<0x0A>", "</s>"]
        
        self.tokenizer = tokenizer
        self.policy_value_sharing = policy_value_sharing

        # usage = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        # self.log_info("Before load_model {}".format(usage))

        base_model = load_model(model_ckpt, tokenizer, init_sft=False, device_map={"":device_rank})
        base_model = prepare_model_for_kbit_training(base_model)

        # usage = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        # self.log_info("Before load policy peft {}".format(usage))

        if rlhf_ckpt is not None:
            rlhf_peft_ckpt = os.path.join(os.path.dirname(rlhf_ckpt), "best_policy_peft")
            self.model = PeftModel.from_pretrained(model=base_model, model_id=rlhf_peft_ckpt, is_trainable=True, adapter_name=policy_adapter_name, device_map={"":device_rank})
        else:
            self.model = PeftModel.from_pretrained(model=base_model, model_id=peft_ckpt, is_trainable=True, adapter_name=policy_adapter_name, device_map={"":device_rank})

        # usage = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        # self.log_info("Before add value peft {}".format(usage))

        # Inference mode defaults to false, meaning that it is trainable
        self.peft_config = get_qlora_config()
        if rlhf_ckpt is not None:
            rlhf_value_peft_ckpt = os.path.join(os.path.dirname(rlhf_ckpt), "best_value_peft")
            self.model.load_adapter(model_id=rlhf_value_peft_ckpt, adapter_name=value_adapter_name, is_trainable=True)
        else:
            self.model.add_adapter(adapter_name=value_adapter_name, peft_config=self.peft_config)


        # usage = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        # self.log_info("Before load reference policy peft {}".format(usage))

        # Set to not trainable so that inference mode is set to true
        self.model.load_adapter(model_id=peft_ckpt, adapter_name=ref_policy_adapter_name, is_trainable=False)
        
        # Adapter names cannot be substring of each other. Otherwise, get_peft_model_state_dict may return state_dict that contains weights of multiple adapters
        self.policy_adapter_name = policy_adapter_name
        self.value_adapter_name = value_adapter_name
        self.ref_policy_adapter_name = ref_policy_adapter_name

        # self.model.print_trainable_parameters()

        usage = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        self.log_info("Before add linear {}".format(usage))

        # regression head for policy-value sharing
        self.linear = torch.nn.Linear(self.model.config.hidden_size, 1).to(self.device)

        if rlhf_ckpt is not None:
            self.linear.load_state_dict(torch.load(rlhf_ckpt)["linear"])

        # TODO: check whether this needs to be changed to self.model.train()
        self.model.eval()

        # self.synced_gpus = self.accelerator.num_processes > 1

    def init_vllm(self):
        usage = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        self.log_info("Before add VLLM {}".format(usage))

        # This should load the VLLM model on card 0
        self.vllm = VLLM(self.model_ckpt, tokenizer_name_or_path=self.peft_ckpt, gpu_memory_utilization=0.37)

        usage = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        self.log_info("After add VLLM".format(usage))


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
        
        prompts_input_ids = prompts_input_ids.to(self.device)

        prompts_attention_mask = prompts_attention_mask.to(self.device)

        prompts_text = self.tokenizer.batch_decode(prompts_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        prompts_input_ids = prompts_input_ids.repeat_interleave(num_return_sequences, dim=0)
        prompts_attention_mask = prompts_attention_mask.repeat_interleave(num_return_sequences, dim=0)
        prompts_len = torch.sum(prompts_attention_mask, dim=1).tolist()


        # self.log_info("Top k: {} | Top p: {}".format(top_k, top_p))
        
        if do_sample:
            sampling_params = SamplingParams(
                n=num_return_sequences,
                max_tokens=self.tokenizer.max_generated_len,
                stop=self.stop_tokens,
                temperature=temperature,
                top_k=top_k if top_k is not None else -1,
                top_p=top_p if top_p is not None else 1.0, # This is commented because top_p is set to null in the configuration
            )
            
        else:
            sampling_params = SamplingParams(
                n=num_return_sequences,
                # best_of=1,
                # use_beam_search=True,
                temperature=0,
                max_tokens=self.tokenizer.max_generated_len,
                stop=self.stop_tokens,
            )

        # self.log_info("Model device before setting policy adapter: {}".format(self.model.device))

        self.model.set_adapter(self.policy_adapter_name)

        # self.log_info("Model device after setting policy adapter: {}".format(self.model.device))

        state_dict = get_peft_model_state_dict(self.model, adapter_name=self.policy_adapter_name)

        state_dict = {k: v.to("cuda:0") for k, v in state_dict.items()}
        lora_merge_unmerge_state_dict(self.vllm.llm, state_dict, self.peft_config, merge=True)
        try:
            generated_text, output = self.vllm.generate(prompts=prompts_text, sampling_params=sampling_params)
        except Exception as e:
            self.log_info(e)
            self.log_info("Restarting VLLM")
            del self.vllm
            self.init_vllm()
            lora_merge_unmerge_state_dict(self.vllm.llm, state_dict, self.peft_config, merge=True)
            generated_text, output = self.vllm.generate(prompts=prompts_text, sampling_params=sampling_params)
        
        lora_merge_unmerge_state_dict(self.vllm.llm, state_dict, self.peft_config, merge=False)

        # self.log_info("prompts_input_ids shape: {}".format(prompts_input_ids.size()))

        combined_input_ids = concat_prompt_and_generation(prompts_input_ids, output, prompts_len, self.tokenizer.max_input_len, self.tokenizer.max_generated_len, self.tokenizer.pad_token_id, self.device)
        # self.log_info("combined_input_ids shape: {}".format(combined_input_ids.size()))

        # begins with 0 ([BOS]); ends with 1 ([EOS])
        combined_attention_mask = (combined_input_ids != self.tokenizer.pad_token_id).long()
        combined_attention_mask_completion_only = combined_attention_mask.clone()
        for i in range(prompts_input_ids.size(0)):
            combined_attention_mask_completion_only[i, :prompts_len[i]] = 0
        
        # generated_input_ids should start with a token in the format _string, and the space should not be a separate token, even if such space is manually added in init sft
        generated_input_ids = remove_prompt_and_pad(combined_input_ids, prompts_len, self.tokenizer.max_generated_len, self.tokenizer.pad_token_id)

        # self.log_info("generated_input_ids shape: {}".format(generated_input_ids.size()))

        generated_attention_mask = (generated_input_ids != self.tokenizer.pad_token_id).long()
        # TODO: Verify that truncated version of generated_input_ids match output extactly and text decoded from generated_input_ids match generated_text exactly
        # generated_text = self.tokenizer.batch_decode(generated_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # repeat input sequences for num_return_sequences times
        prompts_text = [elem for elem in prompts_text for _ in range(num_return_sequences)]

        # print_idx = [0, 1, 4, 5] if num_return_sequences == 4 else [0, 1]
        # if num_return_sequences == 4:
        #     for i in print_idx:
        #         self.log_info("prompt_text {}: {}".format(i, prompts_text[i]))
        #         self.log_info("combined_text {}: {}".format(i, self.tokenizer.decode(combined_input_ids[i])))
        #         self.log_info("generated_text {}: {}".format(i, generated_text[i]))

        # Preprocessing adapted from ALCE. Retain only the first line. Remove end token.
        generated_texts_for_reward = copy.deepcopy(generated_text)
        for i in range(len(generated_texts_for_reward)):
            generated_texts_for_reward[i] = generated_texts_for_reward[i].strip().split("\n")[0] # If too many answers have new lines, then maybe this line needs to be removed. Currently, correctness sequence level reward uses normalized text
            generated_texts_for_reward[i] = generated_texts_for_reward[i].replace("<|im_end|>", "")
        
        # Remove all citations for all non-AutoAIS evaluation
        # normalized_generated_text = copy.deepcopy(generated_texts_for_reward)
        normalized_generated_text = generated_texts_for_reward
        for i in range(len(normalized_generated_text)):
            normalized_generated_text[i] = remove_citations(normalized_generated_text[i])

        return {
            'prompts_text': prompts_text,
            'prompts_input_ids': prompts_input_ids, # (B, input_len)
            'prompts_attention_mask': prompts_attention_mask, # (B, input_len)
            'combined_input_ids': combined_input_ids, # (B, combined_len)
            'combined_attention_mask': combined_attention_mask, # (B, combined_len)
            'combined_attention_mask_completion_only': combined_attention_mask_completion_only, # (B, combined_len)
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
                    #  combined_generated_input_ids: torch.Tensor, # (B, combined_len), no [BOS]
                     generated_input_ids: torch.Tensor, # (B, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len)
                    ):
        self.model.set_adapter(self.policy_adapter_name)
        outputs = self.model(
            input_ids=combined_input_ids, # TODO: Check whether combined_input_ids needs to start with [BOS] or not
            attention_mask=combined_attention_mask,
            labels=mask_pad(combined_input_ids, combined_attention_mask, -100), # The labels for llama are optional and used for mlm only. Do not shift here as shifting will be done in the forward pass according to https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L990
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        combined_logits = outputs.logits # (B, combined_len, V)

        prompts_len_minus_1 = [i-1 for i in prompts_len]

        generated_logits = remove_prompt_and_pad(combined_logits, prompts_len_minus_1, self.tokenizer.max_generated_len, self.tokenizer.pad_token_id) # (B, output_len, V)
        logprobs = F.log_softmax(generated_logits, dim=-1)
        generated_logprobs = torch.gather(logprobs, 2, generated_input_ids[:, :, None]).squeeze(2) # (B, output_len)
        generated_entropy = logits_to_entropy(generated_logits) # (B, output_len)

        results = {
            'generated_logits': generated_logits, # (B, output_len, V)
            'generated_logprobs': mask_pad(generated_logprobs, generated_attention_mask), # (B, output_len)
            'generated_entropy': mask_pad(generated_entropy, generated_attention_mask), # (B, output_len)
        }

        if self.policy_value_sharing:
            hidden_states = outputs.hidden_states[-1] # (B, combined_len, d)
            hidden_states = remove_prompt_and_pad(hidden_states, prompts_len, self.tokenizer.max_generated_len, self.tokenizer.pad_token_id) # (B, output_len, d)
            logits = self.linear(hidden_states).squeeze(-1) # (B, output_len)
            results.update({
                'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, output_len)
            })

        return results

    def ref_p_forward_pass(self,
                     prompts_len: List[int],
                     combined_input_ids: torch.Tensor, # (B, combined_len), includes [BOS]
                     combined_attention_mask: torch.Tensor, # (B, combined_len)
                     generated_input_ids: torch.Tensor, # (B, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len)
                    ):
        self.model.set_adapter(self.ref_policy_adapter_name)
        outputs = self.model(
            input_ids=combined_input_ids, # TODO: Check whether combined_input_ids needs to start with [BOS] or not
            attention_mask=combined_attention_mask,
            labels=mask_pad(combined_input_ids, combined_attention_mask, -100), # The labels for llama are optional and used for mlm only. Do not shift here as shifting will be done in the forward pass according to https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L990
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        combined_logits = outputs.logits # (B, combined_len, V)

        prompts_len_minus_1 = [i-1 for i in prompts_len]

        generated_logits = remove_prompt_and_pad(combined_logits, prompts_len_minus_1, self.tokenizer.max_generated_len, self.tokenizer.pad_token_id) # (B, output_len, V)
        logprobs = F.log_softmax(generated_logits, dim=-1)
        generated_logprobs = torch.gather(logprobs, 2, generated_input_ids[:, :, None]).squeeze(2) # (B, output_len)
        generated_entropy = logits_to_entropy(generated_logits) # (B, output_len)

        results = {
            'generated_logits': generated_logits, # (B, output_len, V)
            'generated_logprobs': mask_pad(generated_logprobs, generated_attention_mask), # (B, output_len)
            'generated_entropy': mask_pad(generated_entropy, generated_attention_mask), # (B, output_len)
        }

        return results



    def value_forward_pass(self,
                     prompts_len: List[int],
                     combined_input_ids: torch.Tensor, # (B, combined_len), include [BOS]
                     combined_attention_mask: torch.Tensor, # (B, combined_len)
                     generated_input_ids: torch.Tensor, # (B, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len)
                    ):
        self.model.set_adapter(self.value_adapter_name)
        outputs = self.model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            labels=mask_pad(combined_input_ids, combined_attention_mask, -100),
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
