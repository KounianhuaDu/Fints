device=3

dataset=abstract_generation
# data_name=caa_python_${dataset}_0.15_qwen3_others
base_data_name=caa_python_${dataset}_0.2_qwen3_others_3
model_name_or_path=/ext0/jxliu/models/Meta-Llama-3.1-8B-Instruct
layer=20
act_location=attnmlp
multipliers=1
alpha=1
beta=1
k=3
cluster=-1
algo=rag # For input-aware steering, set 'algo=PASteer'
vec_strategy=mean
token_range=response
vector_root=./pa_back/caa_data/caa_vector_pt/llama-3.1_${base_data_name}_${act_location}_${vec_strategy}_${token_range}

# for token_range in response all prompt; do
# for percent in 0.05 0.15 0.25 0.5; do
    data_name=${base_data_name}
#     if [ "$algo" != "PASteer" ]; then
#         # for act_location in attn mlp; do
#         vector_root=./pa_back/caa_data/caa_vector_pt/llama-3.1_${data_name}_${act_location}_${vec_strategy}_${token_range}
#         CUDA_VISIBLE_DEVICES=${device} python ./train.py \
#             --layers ${layer} \
#             --data_name ${data_name} \
#             --model_name_or_path ${model_name_or_path} \
#             --act_location ${act_location} \
#             --vec_strategy ${vec_strategy} \
#             --token_range ${token_range} \
#                 # --rerun
#     fi

# act_location=attnmlp
for idx in 3 ; do
    vector_root=./pa_back/caa_data/caa_vector_pt/llama-3.1_${dataset}
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
        --idx ${idx}\
        --token_range ${token_range} \
        # --steering \
        # --eval
done
