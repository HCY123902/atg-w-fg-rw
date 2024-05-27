<h1 align="center"> Training Language Models to Generate Text with Citations via Fine-grained Rewards</h1>

This repository provides code for the paper: Training Language Models to Generate Text with Citations via Fine-grained Rewards. The training scripts are adapted from [FineGrainedRLHF](https://github.com/allenai/FineGrainedRLHF) while the evaluation script is adapted from [ALCE](https://github.com/princeton-nlp/ALCE).

## Content
1. [Set Up](#set-up)
1. [Data](#data)
1. [ChatGPT Distillation](#chatgpt-distillation)
1. [RL](#rl)
      * [Fine-Grained RL](#fine-grained-rl)
      * [Holistic RL](#holistic-rl)
1. [RS](#rs)
      * [Fine-Grained RS](#fine-grained-rs)
      * [Holistic RS](#holistic-rs)
1. [Chaining RS and RL](#chaining-rs-and-rl)
1. [EXPERTQA](#expertqa)
1. [Our Trained Models](#our-trained-models)

## Set Up
```bash
# create a conda environment with python 3.9
conda create --name atg-w-fg-rw python=3.9
conda activate atg-w-fg-rw 

# git clone and install packages
git clone https://github.com/HCY123902/atg-w-fg-rw.git
cd atg-w-fg-rw
pip install -e .
python -m spacy download en_core_web_sm
mkdir tasks/qa_feedback/model_outputs
```

* We use wandb for visualization. Configure your {WANDB_API_KEY} and optionally other settings in `config.sh` and then run
```
bash config.sh
```

* Make sure that you have access to the model weights of LLaMA-2-7B on HuggingFace, and then run
```
huggingface-cli login
```

## Data

Our data can be found [here](https://drive.google.com/drive/folders/16YihOAL6sIytNCKtrV7rU8SXAv8hmJVH?usp=drive_link). We provide the train/dev/test splits for the combined mixture used to train our main model (`combined_{split}.json`), as well as the EXPERTQA examples used exclusively for inference (`expertqa_test.json`).

We also provide the train/dev/test examples for ASQA/QAMPARI/ELI5 used in the *separate* setting (`{dataset}_{split}.json`).

You will need to put these JSON files in `tasks/qa_feedback/data`.

## ChatGPT Distillation
We use the combined mixture for distillation and subsequent training by default. You can also customize the `dataset` option on which dataset to use in the bash script.

**Training**: Run
```bash
bash tasks/qa_feedback/training/distillation.sh
```

**Inference and evaluation**: Open `tasks/qa_feedback/inference/inference.sh` and
1. Replace `{RUN_LOG_NAME}` with `distillation` (or any other name you want)
1. Replace `{PEFT_CKPT_DIR}` with `distillation` (the directory containing the checkpoint)
1. Replace `{TOKENIZER_DIR}` with `distillation` (the same directory containing the tokenizer)
1. Replace `{EVAL_SAVE_DIR}` with `distillation_evaluation` (the directory where inference results are saved)

After that, run
```bash
bash tasks/qa_feedback/inference/inference.sh
```

The metrics results for each dataset will be shown in `tasks/qa_feedback/model_outputs/{EVAL_SAVE_DIR}/{dataset}_result.json.score`.

## RL
You will first need to complete the [distillation](#chatgpt-distillation) step and have a distilled checkpoint in `tasks/qa_feedback/model_outputs/distillation`

### Fine-Grained RL
We provide the default RL configuration at `tasks/qa_feedback/training/rl_fg_config.yml`. You will need at least 40 GB of VRAM to do RL training.

**Training**: Run
```bash
bash tasks/qa_feedback/training/rl_fg.sh
```
to train LLaMA-2-7B with fine-grained rewards, on top of the distilled checkpoint.

**Inference and evaluation**: Configure and run `tasks/qa_feedback/inference/inference.sh`. Refer to (#chatgpt-distillation) but replace each occurance of `distillation` with `rl-fg` when configuring `inference.sh`. There is an exception for `{TOKENIZER_DIR}`, which you should keep as `distillation`, since the same tokenizer is used across different settings.

Note, importantly, that you will also need to make `{PEFT_CKPT_DIR}` in `inference.sh` point to `rl-fg/best_peft/policy_adapter` instead of just `rl-fg`, since the adapter weights are stored in a different place in RL setting.

### Holistic RL
If you want to train with holistic rewards, change `holistic` in `tasks/qa_feedback/training/rl_fg_config.yml` to `True`, and remember to change the checkpoint save path `save_dir` accordingly.

## RS
You will first need to complete the [distillation](#chatgpt-distillation) step and have a distilled checkpoint in `tasks/qa_feedback/model_outputs/distillation`

### Fine-Grained RS
**Sampling**: Run
```
bash tasks/qa_feedback/inference/generate_beam_search.sh
```

This will generate a new JSON file `combined_train_rs_fg.json` at `tasks/qa_feedback/data`, that contains the best sampled sequence for each question.

**Training**: Train on top of the distilled checkpoint at `tasks/qa_feedback/model_outputs/distillation` with `combined_train_rs_fg.json`. This is done by running
```
bash tasks/qa_feedback/training/rs_fg_sft.sh
```

**Inference and evaluation**: Configure and run `tasks/qa_feedback/inference/inference.sh`. Refer to (#chatgpt-distillation) but replace each occurance of `distillation` with `rs-fg-sft` when configuring `inference.sh`. There is an exception for `{TOKENIZER_DIR}`, which you should keep as `distillation`, since the same tokenizer is used across different settings.

### Holistic RS
If you want to train with holistic rewards, run
```
bash tasks/qa_feedback/inference/generate_best_of_n.sh
```
to sample the best responses instead.

## Chaining RS and RL

To chain RL after RS, you will first need to complete the [distillation](#chatgpt-distillation) and [RS](#rs) steps, and have a fine-tuned checkpoint in `tasks/qa_feedback/model_outputs/rs-fg-sft`

You can then change `peft_ckpt` in `tasks/qa_feedback/training/rl_fg_config.yml` to `tasks/qa_feedback/model_outputs/rs-fg-sft` and run
```bash
bash tasks/qa_feedback/training/rl_fg.sh
```

## EXPERTQA

To evaluate our model with EXPERTQA samples, set `--dataset` in `inference.sh` to `expertqa` and then run the script. The model's generation will be stored in `tasks/qa_feedback/model_outputs/{YOUR_CKPT}_evaluation/expertqa_result.json`. You can then use the original [respository](https://github.com/chaitanyamalaviya/expertqa) of EXPERTQA to evaluate the model's generation.

## Our Trained Models

We provide our $\mathcal{M}_{dist}$, $h.RL$, $fg.RL$, $h.RS$, $fg.RS$, $h.(RS+RL)$, and $fg.(RS+RL)$ checkpoints trained in the *combined* setting in [this link](https://drive.google.com/drive/folders/1qNgBgKz9BBor1ra9hHG6WOqAj9UZbYxP?usp=drive_link). Unzip the folders and place them in `tasks/qa_feedback/model_outputs`. These are adapter weights, and you will need to merge them into the original model weights to get a complete model for inference or further training. The following is an example script to do the merge

```python
from transformers import AutoModelForCausalLM
from peft import PeftConfig, PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
peft_ckpt = "tasks/qa_feedback/model_outputs/rl-fg/best_peft/policy_adapter"
peft_model = PeftModel.from_pretrained(base_model, peft_ckpt, is_trainable=True)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("tasks/qa_feedback/model_outputs/rl-fg/merged_with_root_peft")
```

Alternatively, if you want to train the model further, then you can just use `peft_model` in the above script.

## Citation
```
@misc{huang2024training,
      title={Training Language Models to Generate Text with Citations via Fine-grained Rewards}, 
      author={Huang, Chengyu and Wu, Zeqiu and Hu, Yushi and Wang, Wenya},
      journel={arXiv preprint arXiv:2402.04315},
      year={2024}
}
```
