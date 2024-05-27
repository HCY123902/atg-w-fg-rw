# A different max_depth is used on QAMPARI, which is why the script is run separately for each dataset here 
export CUDA_VISIBLE_DEVICES="0,1"
conda activate atg-w-fg-rw
export RUN_LOG_PATH="rs_fg_sampling_asqa.log"
python ./tasks/qa_feedback/inference/generate_beam_search.py \
    --peft_ckpt ./tasks/qa_feedback/model_outputs/distillation \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset asqa \
    --num_return_sequences 2 \
    --top_k 50 \
    --top_p 1.0 \
    --temperature 1.0 \
    --beam_size 8 \
    --max_depth 5
export RUN_LOG_PATH="rs_fg_sampling_qampari.log"
python ./tasks/qa_feedback/inference/generate_beam_search.py \
    --peft_ckpt ./tasks/qa_feedback/model_outputs/distillation \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset qampari \
    --num_return_sequences 2 \
    --top_k 50 \
    --top_p 1.0 \
    --temperature 1.0 \
    --beam_size 8 \
    --max_depth 10 \
    --init_prob 0.5
export RUN_LOG_PATH="rs_fg_sampling_eli5.log"
python ./tasks/qa_feedback/inference/generate_beam_search.py \
    --peft_ckpt ./tasks/qa_feedback/model_outputs/distillation \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset eli5 \
    --num_return_sequences 2 \
    --top_k 50 \
    --top_p 1.0 \
    --temperature 1.0 \
    --beam_size 8 \
    --max_depth 5
python ./tasks/qa_feedback/data/combine_samples.py