export CUDA_VISIBLE_DEVICES=1
MODEL_PATH="/inspire/hdd/global_user/zhangweinan-24046/Meta-Llama-3.1-8B-Instruct"
TASK_NAME="abstract_generation"

K=0
remain_rate=-1
cluster=2


python ./task_LoRA.py \
    --task_name "$TASK_NAME" \
    --llama_model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH"

# Clustering training data. Only run once for each dataset
python ./anchor_selection/history_anchor.py \
    --candidate_path  "../pa_back/data/${TASK_NAME}/processed/train.pkl" \
    --task_name "$TASK_NAME" \
    --k 50

python ./train_anchor_PEFT.py \
    --task_name "$TASK_NAME" \
    --llama_model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --lora_ckpt "./output/${TASK_NAME}/task-base_LLM/lora_ckpt.pt" \
    --output_dir "./output/${TASK_NAME}/Anchor_PEFT_base/LoRA" \
    --k "${K}" \
    --anchor_path "./anchor_selection/${TASK_NAME}/anchor_user_idx.pt" \
    --remain_rate ${remain_rate}

python train_anchor_gate.py \
    --task_name "$TASK_NAME" \
    --llama_model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --lora_ckpt "./output/${TASK_NAME}/task-base_LLM/lora_ckpt.pt" \
    --anchor_path "./output/${TASK_NAME}/Anchor_PEFT_base/LoRA" \
    --anchor_idx_path "./anchor_selection/${TASK_NAME}/anchor_user_idx.pt" \
    --output_dir "./output/${TASK_NAME}/Anchor_PEFT_base/gate" \
    --k "${K}" \
    --remain_rate ${remain_rate}



python lora_composition.py \
    --task_name "$TASK_NAME" \
    --llama_model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --output_dir "./output/${TASK_NAME}/LoRA-Composition" \
    --lora_ckpt "./output/${TASK_NAME}/task-base_LLM/lora_ckpt.pt" \
    --gate_dir "./output/${TASK_NAME}/Anchor_PEFT_base/gate"\
    --anchor_dir "./output/${TASK_NAME}/Anchor_PEFT_base/LoRA" \
    --k "${K}" \
    --remain_rate ${remain_rate} \
    --cluster ${cluster}
    # --distinct 
