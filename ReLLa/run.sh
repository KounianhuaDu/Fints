model=/inspire/hdd/global_user/zhangweinan-24046/Meta-Llama-3.1-8B-Instruct
dataset=abstract_generation
K=3
epochs=5
percent=0.5

for percent in 0.05; do
    torchrun --nproc_per_node 4 --master_port 15637 new_finetune.py \
    --lr 0.001 \
    --dataset ${dataset} \
    --train_size 8192 \
    --train_type sequential \
    --test_type sequential \
    --K ${K} \
    --epochs ${epochs} \
    --total_batch_size 16 \
    --output_path ${dataset}-Meta-Llama-3.1-8B-Instruct/lr_0.001_shot_8192_sequential_sequential_K_${K}_${epochs}_bs256_${percent} \
    --test_range all \
    --model_path ${model} \
    --percent ${percent}
done
