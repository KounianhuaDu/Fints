dataset=pwab_pos
model_weights=/inspire/hdd/global_user/zhangweinan-24046
k=5
llm=llama-3.1
form=json
data_path=../pa_back/data


python data_collect_para.py \
    --dataset ${dataset} \
    --modelweight ${model_weights} \
    --k ${k} \
    --llm ${llm} \
    --form ${form} \
    --data_path ${data_path}

