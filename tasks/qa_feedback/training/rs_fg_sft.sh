export CUDA_VISIBLE_DEVICES="0"
export INIT_SFT_LOG_PATH="rs_fg_sft.log"
export PRED_RUN_NAME="2"
export INIT_RUN_NAME="2"
model_save_path=./tasks/qa_feedback/model_outputs/rs-fg-sft
conda activate atg-w-fg-rw
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./sft/run_llama_sft.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft_ckpt_path ./tasks/qa_feedback/model_outputs/distillation \
    --do_train \
    --do_eval \
    --bf16 \
    --num_train_epochs 3 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --train_file ./tasks/qa_feedback/data/combined_train_rs_fg.json \
    --validation_file ./tasks/qa_feedback/data/combined_dev.json \
    --output_dir $model_save_path \
    --overwrite_output_dir \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=4 \
    --generation_max_length 200 \
    --save_total_limit 12 \
    --load_best_model_at_end \
    --report_to wandb \
    --log_level info \
    --quantize False \
    --mask_input True \
    --use_adapter True \
    --run_name rs_fg_sft