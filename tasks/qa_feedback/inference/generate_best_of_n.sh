export CUDA_VISIBLE_DEVICES="0"
export RUN_LOG_PATH="rs_h_sampling_combined.log"
conda activate atg-w-fg-rw
python ./tasks/qa_feedback/inference/generate_best_of_n_vllm.py \
    --peft_ckpt ./tasks/qa_feedback/model_outputs/distillation \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset combined \
    --num_return_sequences 16 \
    --top_k 50 \
    --top_p 1.0 \
    --temperature 1.0 \
    --batch_size 4