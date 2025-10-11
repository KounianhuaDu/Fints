device=0

dataset=LaMP_4
data_name=caa_python_${dataset}_0.15_qwen3_others
model_name_or_path=/inspire/hdd/global_user/zhangweinan-24046/Meta-Llama-3.1-8B-Instruct
layer=23
act_location=none
multipliers=1
alpha=1
beta=1
k=5
cluster=2
algo=rag # For input-aware steering, set 'algo=PASteer'
vector_root=../pa_back/caa_data/caa_vector_pt/llama-3.1_${data_name}_${act_location}

if [ "$algo" != "PASteer" ]; then
    CUDA_VISIBLE_DEVICES=${device} python ./train.py \
        --layers ${layer} \
        --data_name ${data_name} \
        --model_name_or_path ${model_name_or_path} \
        --act_location ${act_location}
        # --rerun
fi

CUDA_VISIBLE_DEVICES=${device} python ./run_generation.py \
    --dataset ${dataset} \
    --data_name ${data_name} \
    --layers ${layer} \
    --multipliers ${multipliers} \
    --alpha ${alpha} \
    --beta ${beta} \
    --act_location ${act_location} \
    --vector_root ${vector_root} \
    --arch llama3-8b \
    --algo ${algo} \
    --k ${k} \
    --form raw \
    --cluster ${cluster} \
    --steering
    # --eval
