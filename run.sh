device=1

dataset=LaMP_4
data_name=caa_python_${dataset}_0.15_qwen3_others
model_name_or_path=/ext0/jxliu/models/Meta-Llama-3.1-8B-Instruct
layer=23
act_location=whole
multipliers=1
alpha=1
beta=1
k=5
cluster=-1
algo=rag # For input-aware steering, set 'algo=PASteer'
vec_strategy=logistic
token_range=response

# Generate steering vectors
vector_root=./pa_back/caa_data/caa_vector_pt/llama-3.1_${data_name}_${act_location}_${vec_strategy}_${token_range}
if [ "$algo" != "PASteer" ]; then
    CUDA_VISIBLE_DEVICES=${device} python ./train.py \
        --layers ${layer} \
        --data_name ${data_name}\
        --model_name_or_path ${model_name_or_path} \
        --act_location ${act_location} \
        --vec_strategy ${vec_strategy} \
        --token_range ${token_range} \
        # --rerun
fi

# Run generation with or without steering
CUDA_VISIBLE_DEVICES=${device} python ./run_generation.py \
    --dataset ${dataset} \
    --data_name ${data_name} \
    --layers ${layer} \
    --multipliers ${multipliers} \
    --alpha ${alpha} \
    --beta ${beta} \
    --act_location ${act_location} \
    --vector_root ${vector_root} \
    --arch llama_steer\
    --algo ${algo} \
    --k ${k} \
    --form raw \
    --cluster ${cluster} \
    --idx 0 \
    --token_range ${token_range} \
    --steering \
    # --eval
