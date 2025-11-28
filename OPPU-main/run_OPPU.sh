export CUDA_VISIBLE_DEVICES=1
MODEL_PATH="/inspire/hdd/global_user/zhangweinan-24046/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
K=0
TASK_NAME="news_headline"
remain_rate=-1
cluster=-1

# cd ../OPPU-main
# python ./task_LoRA.py --k "$K" --task_name "$TASK_NAME" --model_name "$MODEL_PATH"

# for remain_rate in 0.15 0.05; do
for cluster in 0 1 2; do
    python ./OPPU.py \
        --k "$K" \
        --task_name "$TASK_NAME" \
        --model_name "$MODEL_PATH" \
        --task_lora "./ckpt/${TASK_NAME}/k0-${MODEL_NAME}-task_LoRA_ckpt" \
        --remain_rate $remain_rate \
        --cluster ${cluster}
done
# cd ../PersonalAgent