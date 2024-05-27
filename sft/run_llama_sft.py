"""
Training script for supervised fine-tuning.

Adapted from
https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py
"""

import logging
import os
import sys
import json
from tabnanny import check
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  
import numpy as np
from datasets import load_dataset, Dataset

import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_offline_mode

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from peft.tuners.lora import LoraLayer

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import argparse
from subprocess import check_output

from fgrlhf.utils import load_model, get_qlora_config, get_chat_prompt_from_promt_without_demo

from tqdm import tqdm

import torch

log_file_path = os.environ["INIT_SFT_LOG_PATH"]

log_fh = open(log_file_path, 'a')

sys.stderr = log_fh
sys.stdout = log_fh

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.FileHandler(log_file_path, mode='a')],
)
logger = logging.getLogger(__name__)
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_PROJECT"] = "SFT"
os.environ["WANDB__SERVICE_WAIT"] = "300"
pred_run_name = os.environ["PRED_RUN_NAME"]
init_run_name = os.environ["INIT_RUN_NAME"]


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    peft_ckpt_path: Optional[str] = field(
        default=None, metadata={"help": "Path to load peft weights"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    do_sample: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to do sampling during generation."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    quantize: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to use 4 bit quantization when finetuning the model. Setting to True using QLoRA while setting to False means using LoRA only."
            )
        }
    )
    mask_input: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to mask model input."
            )
        }
    )
    use_adapter: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to use adapter or train the entire model"
            )
        }
    )
    train_device_map: Optional[str] = field(
        default="cuda:0",
        metadata={
            "help": (
                "The train device map"
            )
        }   
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    generation_max_length: Optional[int] = field(
        default=200,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    dataset: Optional[str] = field(
        default="asqa",
        metadata={"help": "The name of dataset to be used."},
    )

    predict_start: Optional[int] = field(
        default=0,
        metadata={"help": "The start prediction index."},
    )

    predict_end: Optional[int] = field(
        default=-1,
        metadata={"help": "The end prediction index."},
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a training or validation file.")

def main():
    # See all possible arguments in src/transformers/training_args.py

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)



    raw_datasets = {}
    if data_args.train_file is not None:
        raw_datasets["train"] = load_dataset("json", data_files=data_args.train_file, split="train")
    if data_args.validation_file is not None:
        raw_datasets["validation"] = load_dataset("json", data_files=data_args.validation_file, split="train")
    if data_args.test_file is not None:
        raw_datasets["test"] = load_dataset("json", data_files=data_args.test_file, split="train")


    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision
    )

    tokenizer.add_special_tokens({"pad_token":"<pad>"}) # Alternative: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'        

    if training_args.do_predict:
        model = load_model(model_args.model_name_or_path, tokenizer, init_sft=True, config=config, model_revision=model_args.model_revision, inference=True, quantize=False, device_map="auto")
    else:
        if not model_args.quantize:
            logger.info("Disabling 4 bit quantization")
        model = load_model(model_args.model_name_or_path, tokenizer, init_sft=True, config=config, model_revision=model_args.model_revision, quantize=model_args.quantize, device_map=model_args.train_device_map)

    



    # Preprocessing the datasets.
    if training_args.do_train:
        train_column_names = raw_datasets["train"].column_names
        valid_column_names = raw_datasets["validation"].column_names
    elif training_args.do_eval:
        valid_column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        test_column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return


    # max_target_length = data_args.max_seq_length
    padding = "max_length" if data_args.pad_to_max_length else False


    def preprocess_function(examples):
        inputs = []
        if "chat" in model_args.model_name_or_path:
            # default_inst = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."

            # Add </s>, according to notebooks using chat verion such as https://colab.research.google.com/drive/134o_cXcMe_lsvl15ZE_4Y75Kstepsntu?usp=sharing#scrollTo=ib_We3NLtj2E and https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing#scrollTo=nAMzy_0FtaUZ
            for i in range(len(examples['prompt_without_demo'])):

                input = get_chat_prompt_from_promt_without_demo(prompt_without_demo=examples['prompt_without_demo'][i], add_answer=True, answer=examples["output"][i][0])

                inputs.append(input)
                if i == 1:
                    logger.info("Original prompt: {}".format(examples['prompt_without_demo'][i]))
                    # logger.info("Question and documents: {}".format(ques_and_doc))
                    logger.info("Completed template: {}".format(input))

            # response_template = "[/INST]"
            # response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        else:
            # remove pairs where at least one record is None
            # Do not add </s>, according to notebooks using base version such as https://colab.research.google.com/drive/1SYpgFpcmtIUzdE7pxqknrM4ArCASfkFQ?usp=sharing#scrollTo=BE7djCZ2_Qdf
            for i in range(len(examples['prompt_without_demo'])):
                if not examples['prompt_without_demo'][i].endswith((' ', '\n', '\t')) and not examples["output"][i][0].startswith((' ', '\n', '\t')):
                    # Different from ALCE as it does not include the whitespace " " in between
                    # In context prompt is now slightly different with the init sft prompt in the sense that this space is added in init sft
                    inputs.append(examples['prompt_without_demo'][i] + " " + examples["output"][i][0] + "</s>")
                else:
                    inputs.append(examples['prompt_without_demo'][i] + examples["output"][i][0] + "</s>")

            # response_template = "\n\nAnswer:"
            # response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[1:]
        model_inputs = tokenizer(inputs, max_length=data_args.max_seq_length, padding=padding, truncation=True)

        # Will not inject response templates when the sequence is truncated, as the answers are appended after the response template here

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            print("Replacing the padding tokens with -100")
            model_inputs["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in model_input] for model_input in model_inputs["input_ids"]
            ]
        
        return model_inputs

    def preprocess_function_predict(examples):
        inputs = []
        # remove pairs where at least one record is None
        for i in range(len(examples['prompt_without_demo'])):
            inputs.append(examples['prompt_without_demo'][i])

        model_inputs = tokenizer(inputs, max_length=data_args.max_seq_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            print("Replacing the padding tokens with -100")
            model_inputs["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in model_input] for model_input in model_inputs["input_ids"]
            ]
        
        return model_inputs

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=valid_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_predict,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=test_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    if model_args.mask_input:
        logger.info("Adopting DataCollatorForCompletionOnlyLM to maks the input")
        # Data collator
        if "chat" in model_args.model_name_or_path:
            response_template = "[/INST]"
            # "[/INST]" -> [518, 29914, 25580, 29962]:['▁[', '/', 'INST', ']']
            response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        else:
            response_template = "\n\nAnswer:"
            # "\n\nAnswer:" -> [29871, 13, 13, 22550, 29901]:['▁', '<0x0A>', '<0x0A>', 'Answer', ':'] -> [13, 13, 22550, 29901]:['<0x0A>', '<0x0A>', 'Answer', ':']
            # Should not use "Answer:" -> ['▁Answer', ':']:[673, 29901] as 673 is different from 22550
            response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[1:]
        # Set the prompt tokens and padding tokens in labels to -100 by default
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, 
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if training_args.fp16 else None
        )
    else:
        # Set the padding tokens in labels to -100 by default
        data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # decoded_preds = []
        anomaly_count = 0
        for i in range(len(preds)):
            if -100 in preds[i]:
                anomaly_count = anomaly_count + 1
        logger.info("{} has token id -100".format(anomaly_count))
        print("{} has token id -100".format(anomaly_count))
        
        if data_args.ignore_pad_token_for_loss:
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        if 2 > 1:
            raise Exception("compute_metrics samples 1: pred: {}; label: {}; anomaly_count: {}".format(decoded_preds[0], decoded_labels[0], anomaly_count))

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    # TODO: Check whether to use fp 16 or bf16 for both training_args and bnb_4bit_compute_type
    # assert not training_args.fp16

    def formatting_prompts_func(example):
        # No whitespace in between
        return example['prompt_without_demo'] + example["output"][0]

    training_arguments = TrainingArguments(
        do_eval=training_args.do_eval,
        num_train_epochs=training_args.num_train_epochs,
        evaluation_strategy=training_args.evaluation_strategy,
        save_strategy=training_args.save_strategy,
        output_dir=training_args.output_dir,
        overwrite_output_dir=training_args.overwrite_output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        save_total_limit=training_args.save_total_limit,
        load_best_model_at_end=training_args.load_best_model_at_end,
        report_to=training_args.report_to,
        metric_for_best_model=training_args.metric_for_best_model,
        log_level="debug",
        optim="paged_adamw_32bit",
        save_steps=training_args.save_steps, #change to 500
        logging_steps=100, #change to 100
        learning_rate=2e-4,
        eval_steps=training_args.eval_steps, #change to 200
        bf16=training_args.bf16,
        fp16=False,
        max_grad_norm=0.3,
        # max_steps=10, #remove this
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=1,
    )

    logger.info(f"Adjusted Training/evaluation parameters {training_arguments}")

    

    # Training
    if training_args.do_train:
        usage = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        logger.info(usage)
        model.gradient_checkpointing_enable()
        if model_args.use_adapter:
            model = prepare_model_for_kbit_training(model)
            if model_args.peft_ckpt_path is not None:
                # print(peft_config)
                # print("Is trainable: {}".format(not peft_config.inference_mode))

                print("Continuing training with the adapter from {}".format(model_args.peft_ckpt_path))
                model = PeftModel.from_pretrained(model, model_args.peft_ckpt_path, is_trainable=True)
            else:
                peft_config = get_qlora_config()

                model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            logger.info("Disabling adapter")

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            data_collator=data_collator,
            compute_metrics=None
        )

        trainer.train()

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()


        tokenizer.save_pretrained(training_args.output_dir)

        usage = check_output(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        logger.info(usage)



    # Evaluation
    max_length = data_args.generation_max_length
        
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams


    if training_args.do_predict:
        logger.info("*** Predict ***")

        model = PeftModel.from_pretrained(model, model_args.peft_ckpt_path, is_trainable=False)
        model.print_trainable_parameters()

        logger.info("Currently used device: {}".format(model.device))



        from transformers import pipeline

        with open(data_args.test_file, "r") as src_json:
            samples = json.load(src_json)
            predict_start = data_args.predict_start
            predict_end = data_args.predict_end if data_args.predict_end >= 0 else len(samples)
            samples = samples[predict_start:predict_end]

        
        pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )

        queries = [sample["prompt_without_demo"] for sample in samples]

        predict_dataset = ListDataset(queries)
        queries_dataloader = torch.utils.data.DataLoader(predict_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False, drop_last=False)

        predictions = []

        # stop = [] if stop is None else stop
        stop = list(set(["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [model.config.eos_token_id, tokenizer.eos_token_id]))

        for batch in tqdm(queries_dataloader):
            if model_args.do_sample:
                sequences = pipeline(
                    batch,
                    return_full_text=False,
                    num_return_sequences=1,
                    # eos_token_id=tokenizer.eos_token_id,
                    eos_token_id=stop_token_ids,
                    max_new_tokens=max_length,
                    # early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    top_k=20
                )
            else:
                sequences = pipeline(
                    batch,
                    return_full_text=False,
                    num_return_sequences=4,
                    # eos_token_id=tokenizer.eos_token_id,
                    eos_token_id=stop_token_ids,
                    max_new_tokens=max_length,
                    # early_stopping=True,
                    num_beams=1,
                )

            for seq in sequences:
                pred = seq[0]['generated_text']
                logger.warning("{}{}".format(queries[len(predictions)], pred))
                predictions.append(pred)
                

    
        for i in tqdm(range(len(samples))):
            pred = predictions[i]

            temp = samples[i]["output"][0] if len(samples[i]["output"]) > 0 else ""
            samples[i]["output"] = pred
            samples[i]["gpt_output"] = temp


        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions_{}_{}.txt".format(predict_start, predict_end))
        with open(output_prediction_file, "w") as writer:
            writer.write("\n\n\n".join(predictions))
        
        pred_file = os.path.join(training_args.output_dir, "{}_result_{}_{}.json".format(data_args.dataset, predict_start, predict_end))
        with open(pred_file, "w") as tgt_json:
            json.dump({"data": samples}, tgt_json, indent=4)

    return {}


if __name__ == "__main__":
    main()