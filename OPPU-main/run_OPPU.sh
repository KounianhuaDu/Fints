export CUDA_VISIBLE_DEVICES=1
MODEL_PATH="/ext0/jxliu/models/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
K=0
TASK_NAME="news_headline"
remain_rate=-1
cluster=-1


python ./task_LoRA.py --k "$K" --task_name "$TASK_NAME" --model_name "$MODEL_PATH"


python ./OPPU.py \
    --k "$K" \
    --task_name "$TASK_NAME" \
    --model_name "$MODEL_PATH" \
    --task_lora "./ckpt/${TASK_NAME}/k0-${MODEL_NAME}-task_LoRA_ckpt" \
    --remain_rate $remain_rate \
    --cluster ${cluster}
