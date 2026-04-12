import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
# from transformers import pipeline, BitsAndBytesConfig
import argparse
# from rank_bm25 import BM25Okapi
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
from utils import split_batch, get_first_k_tokens, print_trainable_parameters, name2taskid
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, extract_news_headline, extract_product_review, extract_scholarly_title, extract_tweet_paraphrasing, extract_abstract_generation
import json
from tqdm import tqdm
import pickle

from instruction import get_his


parser = argparse.ArgumentParser(description="Parser for LoRA")
parser.add_argument('--model_name', type=str, default='../model_weights/Meta-Llama-3.1-8B-Instruct')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--k', type=int, default=0)
parser.add_argument('--max_step', type=int, default=5000)
parser.add_argument('--cut_off', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=3)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--task_name', type=str, default='movie_tagging')
parser.add_argument('--add_profile', action='store_true')
parser.add_argument('--access_token', type=str, default=None)

task_name_dict = {
    'news_headline': 'LaMP_4',
    'abstract_generation': 'abstract_generation'
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
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
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


with open(f"../pa_back/data/{task_name}/processed/remain_train.pkl", 'rb') as f:
    train = pickle.load(f)
    train = list(train.values())


# with open(f"./data/{task_name}/user_top_100_history.json", 'r') as f:
#     test_data = json.load(f)

if args.task_name == "movie_tagging":
    extract_article = extract_movie
elif args.task_name == "news_categorize":
    extract_article = extract_news_cat
elif args.task_name == "news_headline":
    extract_article = extract_news_headline
elif args.task_name == "product_rating":
    extract_article = extrat_product_review
elif args.task_name == "scholarly_title":
    extract_article = extract_scholarly_title
elif args.task_name == "tweet_paraphrase":
    extract_article = extrat_tweet_paraphrasing
elif args.task_name == "abstract_generation":
    extract_article = extract_abstract_generation


with open('./prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)


if args.add_profile:
    with open(f'./data/{task_name}/profile_user_100.json', 'r') as f:
        test_profile = json.load(f)

    with open(f'./data/{task_name}/profile_user_others.json', 'r') as f:
        train_profile = json.load(f)


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



# training
from datasets import load_dataset, Dataset
model = get_peft_model(base_model, peft_config)
print_trainable_parameters(model)

pred_all = []
actual = []
train_data = []

for i in tqdm(range(len(train))):
    if args.add_profile:
        profile = train_profile[i]['output']

    for idx, q in enumerate(train[i]):

        if args.task_name != "citation":
            article = get_first_k_tokens(extract_article(q['input']), 768)
            prompt = prompt_template[args.task_name]['prompt'].format(article)
            full_prompt = prompt_template[args.task_name]['full_prompt'].format(get_first_k_tokens(extract_article(q['input']), 768), q['output'])
        
        else:
            question = q['input']
            article = extract_citation_title(question)
            option1, option2 = extract_option(question, 1), extract_option(question, 2)

            prompt = prompt_template[args.task_name]['prompt'].format(article, option1, option2)
            full_prompt = prompt_template[args.task_name]['full_prompt'].format(article, option1, option2, q['output'])

        if k > 0:
            with open(f"../pa_back/data/{task_name}/processed/remain_train_ranked.json", 'r') as f:
                visible_history_list = json.load(f)

            history_string = get_his(task_name, q['user_id'], k)
            prompt = history_string + "\n" + prompt
            full_prompt = history_string + "\n" + full_prompt

        if args.add_profile:
            prompt = profile + "\n" + prompt
            full_prompt = profile + "\n" + full_prompt
        train_data.append(
            {
                "prompt": prompt,
                "full_prompt": full_prompt
            }
        )

# print(train_data)

train_dataset = Dataset.from_list(train_data).select(range(5000))
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

if args.add_profile:
    output_name = "./ckpt/{}/k{}-{}-{}-profile-task_LoRA_ckpt".format(args.task_name, args.k, args.task_name, model_name.split('/')[-1])
else:
    output_name = "./ckpt/{}/k{}-{}-task_LoRA_ckpt".format(args.task_name, args.k, model_name.split('/')[-1])
    
model.save_pretrained(output_name)
