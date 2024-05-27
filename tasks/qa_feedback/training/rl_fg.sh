export CUDA_VISIBLE_DEVICES="0"
export TRAIN_LOG_PATH="rl_fg.log"
conda activate atg-w-fg-rw
python tasks/qa_feedback/training/train_finegrained_vllm.py --config tasks/qa_feedback/training/rl_fg_config.yml
