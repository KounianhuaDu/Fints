import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel
# from transformers import pipeline, BitsAndBytesConfig
import argparse
# from rank_bm25 import BM25Okapi
# from trl import SFTTrainer, DataCollatorForCompletinOnlyLM
import transformers
from utils import split_batch, get_first_k_tokens, print_trainable_parameters, name2taskid
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, extract_news_headline, extract_product_review, extract_scholarly_title, extract_tweet_paraphrasing, extract_abstract_generation
import json
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
import pickle
import os
import ast
from copy import deepcopy
import torch.nn.functional as F

from personalized_lora import PersonalModel
from instruction import get_his, SYS_PROMPT_SINGLE
from pwab import functions, data

functions_dict = {tool.__name__: tool for tool in functions}
init_data = data


parser = argparse.ArgumentParser(description="Parser for LoRA")
parser.add_argument('--model_name', type=str, default='../model_weights/Meta-Llama-3.1-8B-Instruct')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--max_step', type=int, default=5000)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--task_name', type=str, default='movie_tagging')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--task_lora', type=str, default='./ckpt/movie_tagging/k1-movie_tagging-Llama-2-7b-hf-task_LoRA_ckpt')
parser.add_argument('--access_token', type=str, default=None)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--finsih_idx', type=int, default=10000)
parser.add_argument('--remain_rate', type=float, default=-1)
parser.add_argument("--cluster", type=int, default=-1)

task_name_dict = {
    'news_headline': 'LaMP_4',
    'abstract_generation': 'abstract_generation',
    'pwab': 'pwab_pos'
}
args = parser.parse_args()
model_name = args.model_name
task_name = task_name_dict[args.task_name]
batch_size = args.batch_size
k = args.k
# max_step = args.max_step
cutoff_len = args.cut_off
add_eos_token = False
max_epoch = args.max_epoch

# # 4 bit quantization inference  
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
# )

# 8-bit quantization inference
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_8bit_quant_type="nf8",
#     bnb_8bit_compute_dtype=torch.float16,
#     bnb_8bit_use_double_quant=True,
#    max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
# )

# 16-bit quantization inference
# bnb_config = BitsAndBytesConfig(
#     load_in_16bit=True,
#     bnb_16bit_quant_type="bf16",
#     bnb_16bit_compute_dtype=torch.bfloat16,
#     bnb_16bit_use_double_quant=True,
#     max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
# )

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=args.access_token)
# tokenizer.eos_token = "</s>"
# tokenizer.pad_token = '[PAD]'
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    local_files_only=False,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

base_model.config.use_cache = False
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id


from peft import prepare_model_for_kbit_training

base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)



from peft import LoraConfig, get_peft_model 

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"], # , "k_proj", "out_proj"
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = transformers.TrainingArguments(
    output_dir='outputs/',
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim='adamw_torch',
    num_train_epochs=max_epoch,
    save_steps=1e9,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=1e-2,
    bf16=True,
    max_grad_norm=0.3,
    # max_steps=max_step,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type='linear',
    report_to='none',
)

test_data = dict()
if args.remain_rate > 0:
    with open(f"../pa_back/data/{task_name}/processed/train_{args.remain_rate}.pkl", 'rb') as f:
        all_test_data = pickle.load(f) 
else:
    with open(f"../pa_back/data/{task_name}/processed/train.pkl", 'rb') as f:
        all_test_data = pickle.load(f)
for idx, (id, samples) in enumerate(all_test_data.items()):
    if args.start_idx <= idx < args.finsih_idx:
        test_data[id] = samples
  
test_name = f'seen_test_{args.cluster}' if args.cluster >= 0 else 'seen_test'        
with open(f"../pa_back/data/{task_name}/processed/{test_name}.pkl", 'rb') as f:
    data = pickle.load(f)
    problems = []
    for u_id, samples in data.items():
        problems += samples

format_flag = False
if args.task_name == "movie_tagging":
    extract_article = extract_movie
    format_flag = True
elif args.task_name == "news_categorize":
    extract_article = extract_news_cat
    format_flag = True
elif args.task_name == "news_headline":
    extract_article = extract_news_headline
    format_flag = True
elif args.task_name == "product_rating":
    extract_article = extrat_product_review
    format_flag = True
elif args.task_name == "scholarly_title":
    extract_article = extract_scholarly_title
    format_flag = True
elif args.task_name == "tweet_paraphrase":
    extract_article = extrat_tweet_paraphrasing
elif args.task_name == "abstract_generation":
    extract_article = extract_abstract_generation
    format_flag = True
elif args.task_name == 'pwab':
    extract_article = lambda x: x
    format_flag = True

with open('./prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)


if args.add_profile:
    with open(f'./data/{task_name}/profile_user_100.json', 'r') as f:
        test_profile = json.load(f)


def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = data_point['full_prompt']
    tokenized_full_prompt = tokenize(full_prompt)
    # if not train_on_inputs:
    user_prompt = data_point['prompt']
    
    tokenized_user_prompt = tokenize(
        user_prompt, add_eos_token=add_eos_token
    )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]  # could be sped up, probably
    return tokenized_full_prompt

def calculate_reward(task, action, observation):
    res = [0, 0.0]
    if task['type'] == 'search':
        if action == 'search_product_by_query':
            res[0] = 1
        else :
            return res
        target_asin = task['output']['product_info']['parent_asin']
        if isinstance(observation, list):
            for i in range(len(observation)):
                if target_asin in observation[i]:
                    res[1] = 1 - i/len(observation)
                    break

    elif task['type'] == 'recommend':
        if action == 'get_recommendations_by_history':
            res[0] = 1
        else :
            return res
        target_asin = task['output']['product_info']['parent_asin']
        if isinstance(observation, list):
            for i in range(len(observation)):
                if target_asin in observation[i]:
                    res[1] = 1 - i/len(observation)
                    break

    elif task['type'] == 'review':
        if action == 'add_product_review':
            res[0] = 1
        else :
            return res
        if isinstance(observation, dict):
            target_review = task['output']['review']['text']
            agent_review = observation['review']
            similarity = compute_similarity(target_review, agent_review)
            res[1] = similarity
    
    return res

def compute_similarity(target_review, agent_review):
    sim_tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/global_user/zhangweinan-24046/all-MiniLM-L6-v2")
    sim_model = AutoModel.from_pretrained('/inspire/hdd/global_user/zhangweinan-24046/all-MiniLM-L6-v2').to('cuda')
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentences = [target_review, agent_review]

    encoded_input = sim_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    if torch.cuda.is_available():
        encoded_input.to('cuda')

    with torch.no_grad():
        model_output = sim_model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    similarity = F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0).item()
    del model_output
    del sentence_embeddings
    torch.cuda.empty_cache()

    return similarity



# training
from datasets import load_dataset, Dataset
# model = PeftModel.from_pretrained(model=base_model, model_id=args.task_lora, is_trainable=False)
# base_model = model.merge_and_unload()
# print_trainable_parameters(model)


pred_all = []
actual = []
with open(f"../pa_back/data/{task_name}/processed/train_ranked.json", 'r') as f:
    visible_history_list = json.load(f)

pwab_res = {
    "FACC": [], 
    "RACC":{
        "search": [],
        "recommend": [],
        "review": []
    }
}
for user_id, samples in test_data.items():
    train_data = []
    if user_id not in data.keys():
        continue
    model = get_peft_model(base_model, peft_config)
    print_trainable_parameters(model)

    if args.add_profile:
        profile = test_profile[i]['output']
        
    ids = []
    test_samples = []
    for i in range(len(samples)):
        if task_name == 'LaMP_4':
            for s in samples[i]['profile']:
                if s['id'] not in ids:
                    ids.append(s['id'])
                    test_samples.append(s)
        else:
            test_samples.append(samples[i])
    print(f"User {user_id} got {len(test_samples)} samples to train")

    for idx, q in tqdm(enumerate(test_samples)):
        for key, value in q.items():
            q[key] = extract_article(str(q[key]))

        if args.task_name == 'pwab':
            q['tool_call'] = ast.literal_eval(q['output'])['tool_call']  
        
        prompt = prompt_template[args.task_name]['OPPU_input'].format(**q)
        full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**q)

        if k > 0 and format_flag==True:
            history_string = get_his(args.task_name, str(q['id']), k, visible_history_list)
            prompt = history_string + "\n" + prompt
            full_prompt = history_string + "\n" + full_prompt

        if args.add_profile and format_flag == True:
            prompt = profile + "\n" + prompt
            full_prompt = profile + "\n" + full_prompt
        train_data.append(
            {
                "prompt": prompt,
                "full_prompt": full_prompt
            }
        )

    # print(train_data)

    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(generate_and_tokenize_prompt).shuffle()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)


    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    output_name = "./ckpt/{}/k{}-{}-{}-OPPU_LoRA".format(args.task_name, args.k, user_id, model_name.split('/')[-1])
    model.save_pretrained(output_name)

    model.eval()
    model.config.use_cache = True  # silence the warnings. Please re-enable for inference!

    # test inference
    personal_model = PersonalModel(args, model, tokenizer)
    outs = []
    eval_k = 5
    for problem_instance in tqdm(data[user_id]):
        # Generate Code & Trace
        res = personal_model.generate(problem_instance, eval_k)
        if res:
            output_dict = res
        else:
            print(f"Generation Error for problem {problem_instance['id']}.")
            continue
        
        outs.append(output_dict)
        if args.task_name == 'pwab':
            try:
                output_dict['generation'] = json.loads(output_dict['generation'])
                action = output_dict['generation']['tool_call']
                all_data = deepcopy(init_data)
                obs = functions_dict[action["name"]](
                    data=all_data, **action["arguments"]
                )
                res = calculate_reward(problem_instance, action['name'], obs)
            except Exception as e:
                print(f"Error: {e}")
                res = [0, 0.0]
            print(res)
            pwab_res["FACC"].append(res[0])
            pwab_res["RACC"][problem_instance['type']].append(res[1])
            
   
    output_dir = f'./output/{args.task_name}/k{args.k}-{model_name.split("/")[-1]}_{eval_k}_{args.remain_rate}_{args.cluster}'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'output-OPPU-{}.json'.format(user_id)), 'w') as f:
        json.dump(
            {
                "task": args.task_name,
                "golds": outs,
            },
            f,
        )
        
if args.task_name == 'pwab':
    evaluation_res = f'./res/{args.task_name}/k{args.k}-{model_name.split("/")[-1]}_{eval_k}_{args.remain_rate}'
    os.makedirs(evaluation_res, exist_ok=True)
    pwab_res["FACC"] = sum(pwab_res["FACC"]) / len(pwab_res["FACC"])
    for func, acc in pwab_res["RACC"].items():
        pwab_res["RACC"][func] = sum(acc) / max(len(acc), 1)
    with open(os.path.join(evaluation_res, 'res.json'), 'w') as f:
        json.dump(pwab_res, f, indent=4)
    print(pwab_res)
#     if args.add_profile:
#         profile = test_profile[i]['output']

#     if k > 0:
#         visible_history_list = test_data[i]['profile']
#         for p in visible_history_list:
#             for key, value in p.items():
#                 p[key] = get_first_k_tokens(p[key], 368)

#         history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]

#         tokenized_corpus = [doc.split(" ") for doc in history_list]
#         bm25 = BM25Okapi(tokenized_corpus)

#     test_question_list = []
#     question_id_list = []

#     for q in test_data[i]['query']:

#         if args.task_name == 'citation':
#             test_question = q['input']
#             test_article = extract_citation_title(test_question)
#             option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
#             test_prompt = prompt_template[args.task_name]['prompt'].format(test_article, option1, option2)

#         else:
#             test_question = q['input']
#             test_article = extract_article(test_question)
#             test_prompt =  prompt_template[args.task_name]['prompt'].format(test_article)

#         if k > 0:
#             tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_article).split(" ")
#             retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)
        
#             history_string = "".join(retrieved_history)
#             test_prompt = history_string + "\n" + test_prompt

#         if args.add_profile:
#             test_prompt = profile + "\n" + test_prompt

#         test_question_list.append(test_prompt)
#         question_id_list.append(q['id'])

#     test_batch_list = split_batch(test_question_list, 1)
#     out_list = []

#     with torch.no_grad():
#         for batch_idx, batch in tqdm(enumerate(test_batch_list), total=len(test_batch_list)):
#             # try:
#             sentences = batch
#             inputs = tokenizer(sentences, return_tensors="pt", padding=True, return_token_type_ids=False)
#             inputs = inputs.to(model.device)

#             with torch.autocast(device_type="cuda"):
#                 outputs = model.generate(
#                     **inputs,
#                     do_sample=True,
#                     top_k=10,
#                     temperature=args.temperature,
#                     top_p=0.9,
#                     eos_token_id=tokenizer.eos_token_id,
#                     max_new_tokens=200
#                 )

#             out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#             out_list += out_sentence
#             # except:
#             #     out_list += ['']
                
#     for i in range(len(out_list)):
#         output = out_list[i].replace(test_question_list[i], '')
#         pred_all.append({
#             "id": question_id_list[i],
#             "output": output
#             })
        
#         print(output)

# output_file = {
#     'task': name2taskid[args.task_name],
#     'golds': pred_all,
#     'model': model_name,
# }

# if args.add_profile:
#     with open('./output/{}/output-OPPU-k{}-{}-{}-profile.json'.format(args.k, args.task_name, args.task_name, model_name.split('/')[-1]), 'w') as f:
#         json.dump(output_file, f, indent=4)
# else:
#     with open('./output/{}/output-OPPU-k{}-{}-{}.json'.format(args.k, args.task_name, args.task_name, model_name.split('/')[-1]), 'w') as f:
#         json.dump(output_file, f, indent=4)