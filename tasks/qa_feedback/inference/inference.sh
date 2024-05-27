export CUDA_VISIBLE_DEVICES="0"
conda activate atg-w-fg-rw
run_log_name={RUN_LOG_NAME}
export RUN_LOG_PATH="${run_log_name}_inference.log"
eval_save_path=./tasks/qa_feedback/model_outputs/{EVAL_SAVE_DIR}
python ./tasks/qa_feedback/inference/inference_vllm.py \
    --peft_ckpt "./tasks/qa_feedback/model_outputs/{PEFT_CKPT_DIR}" \
    --tokenizer_path "./tasks/qa_feedback/model_outputs/{TOKENIZER_DIR}" \
    --base_model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --save_path $eval_save_path \
    --dataset expertqa
export EVAL_LOG_PATH="${run_log_name}_evaluation_asqa.log"
python ./evaluation/eval.py \
    --f ${eval_save_path}/asqa_result.json \
    --citations \
    --mauve \
    --no_rouge \
    --nli_model_name_or_path google/t5_xxl_true_nli_mixture \
    --nli_model_type bf16 \
    --no_doc_rec_corr
export EVAL_LOG_PATH="${run_log_name}_evaluation_qampari.log"
python ./evaluation/eval.py \
    --f ${eval_save_path}/qampari_result.json \
    --citations \
    --nli_model_name_or_path google/t5_xxl_true_nli_mixture \
    --nli_model_type bf16 \
    --no_doc_rec_corr
export EVAL_LOG_PATH="${run_log_name}_evaluation_eli5.log"
python ./evaluation/eval.py \
    --f ${eval_save_path}/eli5_result.json \
    --citations \
    --claims_nli \
    --mauve \
    --nli_model_name_or_path google/t5_xxl_true_nli_mixture \
    --nli_model_type bf16 \
    --no_doc_rec_corr