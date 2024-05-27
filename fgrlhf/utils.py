# from: https://github.com/liujch1998/rainier/

from genericpath import exists
import json
from lib2to3.pgen2.tokenize import tokenize
from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import collections
import re
import copy

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig

from vllm import LLM, SamplingParams

NEGATIVE_INF = -100000.0

T = TypeVar('T')

def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)


def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)


def reduce_std(value, mask):
    return torch.sqrt(reduce_mean(torch.square(value), mask) - torch.square(reduce_mean(value, mask)))


def reduce_var(value, mask):
    return reduce_mean(torch.square(value), mask) - torch.square(reduce_mean(value, mask))


def logits_to_entropy(logits):
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.entropy()


def mask_pad(value, mask, pad_value=None):
    if pad_value is None:
        pad_value = NEGATIVE_INF
    return value * mask + pad_value * (1 - mask)


def clamp(value, min_value, max_value):
    return torch.max(torch.min(value, max_value), min_value)


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


def whiten(values, masks, shift_mean=True, accelerator=None):
    if accelerator is not None:
        all_values = accelerator.gather(values) # (num_gpus * B, KL)
        all_masks = accelerator.gather(masks) # (num_gpus * B, KL)
        mean, var = reduce_mean(all_values, all_masks), reduce_std(all_values, all_masks)
    else:
        mean, var = reduce_mean(values, masks), reduce_std(values, masks)
    # if accelerator is not None and accelerator.is_main_process:
    #     print(f'all_values: {all_values}, all_masks: {all_masks}')
    #     print(f'mean: {mean}, var: {var}')
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def flatten_dict(nested, sep='.'):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def set_seed(seed=19260817, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)


def args_to_filename(args):
    return f'_reward-{args.reward_shape}'
    '''
    return "_klCoef" + str(args.kl_coef) + \
        "_lr" + str(args.lr) + \
        "_batchSize" + str(args.batch_size) + \
        "_eps" + str(args.total_episodes) + \
        "_temp" + str(args.temperature) + \
        "_initModel_" + str(args.init_model_type) + \
        "_refModel_" + str(args.ref_model_type) + \
        "_valModel_" + str(args.value_model_type) + \
        "_respLen" + str(args.response_length) + \
        "_realKL_" + str(args.real_kl)
    '''

def get_tensorboard_logname(comment=""):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + comment)
    return log_dir

def get_bnb_config(compute_type):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_type # Notebooks use float16
    )

def load_model(model_name_or_path, tokenizer, init_sft=True, config=None, model_revision=None, inference=False, quantize=True, device_map=None):
    compute_type = getattr(torch, ('float16' if inference else "bfloat16")) # Use bf16 in training and fp16 in inference
    bnb_config = get_bnb_config(compute_type)
    if init_sft:
        if quantize:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                quantization_config=bnb_config,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
                revision=model_revision
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=compute_type,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
                revision=model_revision
            )
    else:
        # Do not use device map as the model will be passed to accelerator.prepare for parallelization
        if quantize:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, quantization_config=bnb_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=compute_type)

    #Resize the embeddings
    model.resize_token_embeddings(len(tokenizer))
    #Configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching
    return model

def get_qlora_config():
    # Inference mode defaults to false, meaning that it is trainable
    peft_config = LoraConfig(
        lora_alpha=16, # Alternative: 32
        lora_dropout=0.1,
        r=64, # Alternative: 8
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules= ["q_proj","v_proj"]
    )
    assert not peft_config.inference_mode
    return peft_config

def remove_prompt_and_pad(gen_input_ids_or_logits, prompts_len, max_generated_len, pad_token_id):
    res = []
    num_dim = gen_input_ids_or_logits.dim()
    for i in range(gen_input_ids_or_logits.size(0)):
        if num_dim == 2:
            gen_input_id = gen_input_ids_or_logits[i, max(prompts_len[i], 1):] # no beginning; ends with 1 ([EOS])
            # This is actually truncating the sequence instead of adding padding tokens
            padded_input_id = F.pad(gen_input_id, (0, max_generated_len - gen_input_id.size(0)), value=pad_token_id)
        elif num_dim == 3:
            gen_input_id = gen_input_ids_or_logits[i, max(prompts_len[i], 1):, :] # no beginning; ends with 1 ([EOS])
            # This is actually truncating the sequence instead of adding padding tokens
            padded_input_id = F.pad(gen_input_id, (0, 0, 0, max_generated_len - gen_input_id.size(0)), value=pad_token_id) # Pad the first dimension to output_len only. Do not pad the last dimension with size V or d
        res.append(padded_input_id)

    res = torch.stack(res, dim=0).contiguous()
    return res

def concat_prompt_and_generation(prompts_input_ids, output, prompts_len, max_input_len, max_generated_len, pad_token_id, device):
    combined_input_ids = []
    for i in range(prompts_input_ids.size(0)):
        # Option 1: Remove the padding tokens in between
        # Option 2: Keep the padding tokens in between
        cat_seq = torch.cat((prompts_input_ids[i][:prompts_len[i]], torch.tensor(output[i]).long().to(device)), dim=0)
        padded_seq = F.pad(cat_seq, (0, max_input_len + max_generated_len - cat_seq.size(0)), value=pad_token_id) # (B, output_len)
        combined_input_ids.append(padded_seq)

    combined_input_ids = torch.stack(combined_input_ids, dim=0).contiguous()
    return combined_input_ids

def peft_merge_and_save(peft_model, save_path, tokenizer=None):
    base_model = peft_model.merge_and_unload()
    base_model.save_pretrained(save_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)

def is_abstaining(dataset, pred_text):
    return (("There is no " in pred_text or "There are no " in pred_text) and "answer" in pred_text) or ("It is unclear " in pred_text)

class VLLM:
    def __init__(self, model_name_or_path, tokenizer_name_or_path=None, max_tokens=200, num_return_sequences=4, stop_tokens=["\n", "Ċ", "ĊĊ", "<0x0A>", "</s>"], sample=True, top_k=20, top_p=1.0, temperature=0.7, gpu_memory_utilization=0.9, max_model_len=None): 
        self.llm = LLM(
            model=model_name_or_path,
            tokenizer=tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            max_model_len=max_model_len
        )
        if not sample:
            self.sampling_params = SamplingParams(
                n=1,
                # best_of=1,
                # use_beam_search=True,
                temperature=0,
                max_tokens=max_tokens,
                stop=stop_tokens,
            )
        else:
            self.sampling_params = SamplingParams(
                n=num_return_sequences,
                max_tokens=max_tokens,
                stop=stop_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        
    def generate(self, prompts=None, prompt_token_ids=None, sampling_params=None):
        s = self.sampling_params if sampling_params is None else sampling_params
        if prompt_token_ids is not None:
            outputs = self.llm.generate(sampling_params=s, prompt_token_ids=prompt_token_ids, use_tqdm=False)
        else:
            # print("prompts: {}".format(prompts))
            # print("Sampling params: {}\n\n{}".format(self.sampling_params, s))
            outputs = self.llm.generate(prompts, sampling_params=s, use_tqdm=False)
        return [req_o.text for o in outputs for req_o in o.outputs], [req_o.token_ids for o in outputs for req_o in o.outputs]

    def init_beam_search_rw_m(self, beam_search_rw_m, seed=42):
        self.beam_search_rw_m = beam_search_rw_m
        set_seed(seed, True)

    def sample_sequences(self, item, seq, pred_texts, pred_tokens, sent_sep):
        new_seqs = []
        for pred_text, pred_token in zip(pred_texts, pred_tokens):
            r = self.beam_search_rw_m.get_reward(item, seq, pred_text, seq["hits"])

            assert set(r["new_hits"]).isdisjoint(seq["hits"])

            new_seq = {
                "pred_text": seq["pred_text"] + pred_text,
                "reward": seq["reward"] + r["rw_change"],
                "cit_rec_rw": seq["cit_rec_rw"] + r["cit_rec_rw_change"],
                "cit_prec_rw": seq["cit_prec_rw"] + r["cit_prec_rw_change"],
                "corr_rec_rw": seq["corr_rec_rw"] + r["corr_rec_rw_change"],
                "entail": seq["entail"] + r["entail"],
                "num_sents": seq["num_sents"] + r["num_sents"],
                "entail_prec": seq["entail_prec"] + r["entail_prec"],
                "total_citations": seq["total_citations"] + r["total_citations"],
                "hits": seq["hits"] + r["new_hits"],
                "terminated": 2 in pred_token or is_abstaining(item["dataset"], pred_text)
            }
            if not new_seq["pred_text"].endswith(sent_sep):
                # print("The last 2 tokens: {}".format(pred_token[-2:]))
                # Token 1822 "]." and token 29889 "." commonly appears as the last token in the sequence
                if item["dataset"] != "qampari" or not new_seq["terminated"]:
                    new_seq["pred_text"] = new_seq["pred_text"] + sent_sep
            new_seqs.append(new_seq)

        print("After sampling, new_seqs:\n{}".format("\n".join(str(seq) for seq in new_seqs)))
        return new_seqs

    def generate_beam_search(self, item, beam_size=2, max_depth=5, include_demo=False, init_prob=-1):
        curr_depth = 0
        # node_id = 0
        # prediction_tree = {}
        # levels = {}
        # prediction_tree[node_id] = {"prompt": prompt, "pred": "",
        #                             "processed_pred": "", "score": None, "parent": None}
        # levels[0] = [0]
        sent_sep = "." if item["dataset"] != "qampari" else ","

        # if item["dataset"] != "qampari":
        assert sent_sep in self.sampling_params.stop, self.sampling_params.stop
        # else:
        #     assert "," in self.sampling_params.stop, self.sampling_params.stop
        
        curr_seqs = []

        if include_demo:
            prompt = item["prompt"]
        else:
            prompt = item["prompt_without_demo"]

        while curr_depth < max_depth:
            if len(curr_seqs) > 0 and all([seq["terminated"] for seq in curr_seqs]):
                print("Terminating early as every sequence has terminated")
                break
            print("curr_depth: {}".format(curr_depth))
            print("curr_seqs:\n{}".format("\n".join([str(seq) for seq in curr_seqs])))
            # levels[curr_depth] = []

            if len(curr_seqs) == 0:
                init_sampling_params = SamplingParams(
                    n=beam_size,
                    max_tokens=self.sampling_params.max_tokens,
                    stop=self.sampling_params.stop,
                    temperature=self.sampling_params.temperature,
                    top_k=self.sampling_params.top_k,
                    top_p=self.sampling_params.top_p,
                )
                outputs = self.llm.generate([prompt], sampling_params=init_sampling_params, use_tqdm=False)
                # Prepending instead of extending, so that newer sequences with the same reward will be selected in place of older sequences
                curr_seqs[:0] = self.sample_sequences(
                    item, 
                    {
                        "pred_text": "",
                        "reward": 0,
                        "cit_rec_rw": 0,
                        "cit_prec_rw": 0,
                        "corr_rec_rw": 0,
                        "entail": 0,
                        "num_sents": 0,
                        "entail_prec": 0,
                        "total_citations": 0,
                        "hits": [],
                    },
                    [req_o.text for o in outputs for req_o in o.outputs], 
                    [req_o.token_ids for o in outputs for req_o in o.outputs],
                    sent_sep
                )
            else:
                prompts = []
                non_term_seqs = [(-len(curr_seqs) + idx, sequence) for idx, sequence in enumerate(curr_seqs) if not sequence["terminated"]]
                for offset, seq in non_term_seqs:
                    prompts.append(prompt + seq["pred_text"])
                outputs = self.llm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
                
                if curr_depth < max_depth - 1 and init_prob > 0:
                    pop_idx = []
                    trans_prob = init_prob * ((max_depth - 1 - curr_depth)/(max_depth - 2))
                    print("Poping elements with probability {}".format(trans_prob))

                for i, (offset, seq) in enumerate(non_term_seqs):
                    curr_seqs[:0] = self.sample_sequences(
                        item,
                        seq,
                        [req_o.text for req_o in outputs[i].outputs],
                        [req_o.token_ids for req_o in outputs[i].outputs],
                        sent_sep
                    )
                    if curr_depth < max_depth - 1 and init_prob > 0:
                        if random.random() < trans_prob:
                            pop_idx.append(offset)

                # Randomly forcing the new pred_text to be added
                if curr_depth < max_depth - 1 and init_prob > 0:
                    for offset in pop_idx:
                        print("Poping {} to force a transition: {}".format(offset, curr_seqs.pop(offset)))
            
            print("\n===========================================\n")
            curr_seqs = sorted(curr_seqs, key=lambda t: t["reward"], reverse=True)
            
            curr_seqs = curr_seqs[:beam_size]
            curr_depth = curr_depth + 1
            item["beam_search_depth_{}".format(curr_depth)] = copy.deepcopy(curr_seqs)

        return curr_seqs


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def lora_reassign_weights(model, state_dict, r, lora_alpha, fan_in_fan_out=False, merge=True):
    is_merged = getattr(model, "is_merged", False)
    assert is_merged != merge, f'{is_merged} != {merge}: if is_merged, then must be unmerge; if not is_merged, then must merge'
    named_params = [(n, p) for n, p in model.named_parameters()]
    scaling = lora_alpha / r
    # print(f'Lora configs: alpha={lora_alpha}, r={r}, scaling={scaling}')
    # print("stat_dict keys: {}".format(state_dict.keys()))
    state_dict = {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}
    # print("model param keys: {}".format([n for (n, _) in named_params]))
    replaced = set()
    merged_names = {
        # these are projector weights that got combined into single matrix in vllm
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }
    non_merged_names = ['o_proj', 'down_proj']
    for name, param in named_params:
        param.requires_grad = False
        if "_proj.weight" not in name:
            continue
        for wn, wn_series in merged_names.items():
            if name.endswith(f"{wn}.weight"):
                for stride_id, att_weight_name in enumerate(wn_series):
                    lora_a = name.replace(f"{wn}.weight", f"{att_weight_name}.lora_A.weight")
                    lora_b = name.replace(f"{wn}.weight", f"{att_weight_name}.lora_B.weight")
                    shard_size = param.shape[0] // len(wn_series)
                    if lora_a in state_dict:
                        # print("Replacing part of {} with {} and {}".format(name, lora_a, lora_b))
                        assert lora_b in state_dict, f'{lora_b} not in state_dict'
                        assert state_dict[lora_b].shape[1] == r, f'{r} != {state_dict[lora_b].shape}'
                        matrix = transpose(state_dict[lora_b] @ state_dict[lora_a], fan_in_fan_out) * scaling
                        assert param.data[shard_size * stride_id:shard_size * (stride_id + 1)].shape == matrix.shape
                        if merge:
                            param.data[shard_size * stride_id:shard_size * (stride_id + 1)] += matrix
                        else:
                            param.data[shard_size * stride_id:shard_size * (stride_id + 1)] -= matrix
                        replaced.add(lora_a)
                        replaced.add(lora_b)
        for wn in non_merged_names:
            if name.endswith(f"{wn}.weight"):
                lora_a = name.replace(f"{wn}.weight", f"{wn}.lora_A.weight")
                lora_b = name.replace(f"{wn}.weight", f"{wn}.lora_B.weight")
                if lora_a in state_dict:
                    assert lora_b in state_dict
                    matrix = transpose(state_dict[lora_b] @ state_dict[lora_a], fan_in_fan_out) * scaling
                    assert param.data.shape == matrix.shape, f'invalid shape: {name} {param.data.shape} != {matrix.shape}'
                    if merge:
                        param.data += matrix
                    else:
                        param.data -= matrix
                    replaced.add(lora_a)
                    replaced.add(lora_b)
    no_replaced = [k for k in state_dict.keys() if k not in replaced]
    assert len(no_replaced) == 0, f'some lora states not loaded, check again!: {no_replaced}'
    # assert all([value_adapter_name in k for k in no_replaced]), f'some lora states not loaded, check again!: {no_replaced}'
    model.is_merged = merge


def lora_merge_unmerge_state_dict(llm, state_dict, peft_config, merge=True):
    # merge lora states to weights
    for worker in llm.llm_engine.workers:
        lora_reassign_weights(worker.model, state_dict, 
            r=peft_config.r, 
            lora_alpha=peft_config.lora_alpha, 
            fan_in_fan_out=peft_config.fan_in_fan_out, 
            merge=merge
        )

DEFAULT_INST = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."
# Do not include <s> as this will be added by the tokenizer
# There are discrepancies between the tokenizers of meta-llama/Llama-2-7b-chat-hf and NousResearch/Llama-2-7b-chat-hf
# The tokenizer of meta-llama/Llama-2-7b-chat-hf works the same as meta-llama/Llama-2-7b-hf, in the sense that </s> can be correctly parsed regardless of whether it is preceded by a whitespace
# In addition, the tokenizer will always add an additional whitespace regardless of whether there is an existing whitespace between <s> and the first token
# This means no whitespace between <s> and the first token needs to be added beforehand
# The tokenzier of NousResearch/Llama-2-7b-chat-hf can only parse </s> when it is preceded by a whitespace
# In addition, the tokenizer will only add an additional whitespace when its input prompt does not start with an <s>
CHAT_TEMPLATE = """[INST] <<SYS>>
{your_system_message}
<</SYS>>

{user_message_1} [/INST] {model_reply_1}</s>"""

def get_chat_prompt_from_promt_without_demo(prompt_without_demo, inst=DEFAULT_INST, add_answer=False, answer=""):
    ques_and_doc = get_ques_and_doc_from_prompt(prompt_without_demo, inst=inst)
    input = get_chat_prompt(ques_and_doc, inst=inst, add_answer=add_answer, answer=answer)
    return input

def get_ques_and_doc_from_prompt(prompt_without_demo, inst):
    ques_and_doc = prompt_without_demo[len(inst + "\n\n"):-len("\n\nAnswer:")]
    return ques_and_doc

def get_chat_prompt(ques_and_doc, inst, add_answer, answer=""):
    if add_answer:
        return CHAT_TEMPLATE.replace("{your_system_message}", inst).replace("{user_message_1}", ques_and_doc).replace("{model_reply_1}", answer)

    return CHAT_TEMPLATE.replace("{your_system_message}", inst).replace("{user_message_1}", ques_and_doc).replace(" {model_reply_1}</s>", "")

DEFAULT_DEVICE_RANK = 0
