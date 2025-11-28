import sys 
sys.path.append("..") 

import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path
# from rank_bm25 import BM25Okapi
from functools import partial

import numpy as np
# import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
from engine_finetuning import train_one_epoch, val_one_epoch, load_model, load_generator_from_raw, load_generator_from_trained
from torch.utils.data import Dataset

from utils import split_batch, get_first_k_tokens, name2taskid
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, extract_news_headline, extract_product_review, extract_scholarly_title, extract_tweet_paraphrasing, extract_abstract_generation

# from torch.utils.tensorboard import SummaryWriter
# from util.misc import NativeScalerWithGradNormCount as NativeScaler

from llama import Tokenizer
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
from tqdm import trange, tqdm
import pickle
from transformers import AutoTokenizer, AutoModel
from colorama import Fore, init
init(autoreset=True)
import re
from copy import deepcopy
import torch.nn.functional as F
import io
import contextlib

from instruction import get_his, build_rag_instruction, SYS_PROMPT_SINGLE
from pwab import functions, data

functions_dict = {tool.__name__: tool for tool in functions}
init_data = data


class InstructionDataset(Dataset):
    def __init__(self, data_list, tokenizer_path, max_tokens=2048):
        self.ann = data_list

        self.max_words = max_tokens
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        # return example, labels, example_mask
        ann = self.ann[index]
        prompt = ann['prompt']
        example = ann['full_prompt']

        prompt = torch.tensor(self.tokenizer1.encode(prompt), dtype=torch.int64)
        example = torch.tensor(self.tokenizer1.encode(example), dtype=torch.int64)

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        return example, labels, prompt


def collate_fn(batch, max_length=2048):
    examples, labels, prompts = zip(*batch)
    # Trim sequences to max_length
    trimmed_examples = [example[:max_length] for example in examples]
    trimmed_labels = [label[:max_length] for label in labels]
    
    # Determine the maximum sequence length after trimming but capped at max_length
    max_length = min(max([len(example) for example in trimmed_examples]), max_length)

    # Pad sequences to the determined max_length
    padded_examples = torch.stack([torch.cat((example, torch.zeros(max_length - len(example), dtype=torch.int64) - 1)) if len(example) < max_length else example for example in trimmed_examples])
    padded_labels = torch.stack([torch.cat((label, torch.zeros(max_length - len(label), dtype=torch.int64) - 1)) if len(label) < max_length else label for label in trimmed_labels])

    example_masks = padded_examples.ge(0)
    label_masks = padded_labels.ge(0)

    padded_examples[~example_masks] = 0
    padded_labels[~label_masks] = 0

    example_masks = example_masks.float()
    label_masks = label_masks.float()

    return padded_examples, padded_labels, example_masks


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--warmup_epochs", default=0, type=int)

    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument("--llama_model_path", default="../model_weights/Meta-Llama-3.1-8B-Instruct", type=str, help="path of llama model")
    parser.add_argument("--tokenizer_path", default="../model_weights/Meta-Llama-3.1-8B-Instruct", type=str, help="path of llama model tokenizer")
    
    parser.add_argument("--model", default="llama7B_lora", type=str, metavar="MODEL", help="Name of model to train")

    parser.add_argument("--max_seq_len", type=int, default=3500, metavar="LENGTH", help="the maximum sequence length")
    
    parser.add_argument("--w_lora", type=bool, default=True, help="use lora or not")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (absolute lr)")
    parser.add_argument("--clip", type=float, default=0.3, help="gradient clipping")

    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--task_name", default="news_headline", type=str, metavar="MODEL", help="name of the task")

    # Dataset parameters
    # parser.add_argument("--test_data_path", default="/afs/crc.nd.edu/user/z/ztan3/Private/LoRA-composition/LaMP_data-final/movie/test_100/user_test_100.json", type=str, help="dataset path")
    # parser.add_argument("--train_data_path", default="/afs/crc.nd.edu/user/z/ztan3/Private/LoRA-composition/LaMP_data-final/movie/user_base_LLM.json", type=str, help="dataset path")
    
    parser.add_argument("--output_dir", default="./output/news_headline/LoRA-Composition", help="path where to save, empty for no saving")

    parser.add_argument("--log_dir", default="./output", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lora_ckpt", default='./output/news_headline/task-base_LLM/lora_ckpt.pt', help="resume lora from checkpoint")
    parser.add_argument("--grad_ckpt", type=bool, default=True, help="whether to use gradient checkpoint, recommend TRUE!!")

    parser.add_argument("--gate_dir", default='./output/news_headline/Anchor_PEFT/gate', help="resume lora from checkpoint")
    parser.add_argument("--anchor_dir", default='./output/news_headline/Anchor_PEFT/LoRA', help="resume lora from checkpoint")
    parser.add_argument("--test_idx_dir", default='./anchor_selection/news_headline/anchor_user_idx.pt', help="resume lora from checkpoint")

    # parser.add_argument("--test_dir", default='/afs/crc.nd.edu/user/z/ztan3/Private/LoRA-composition/LaMP_data-final/movie/test_100/user_test_100.json', help="resume lora from checkpoint")


    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # generation hyperparameters
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature")
    parser.add_argument("--max_gen_len", type=int, default=768, help="top_p")

    parser.add_argument("--k", type=int, default=3, help="top_p")
    parser.add_argument('--infer', default=False, action='store_true')


    # lora composition hyperparameters
    parser.add_argument("--topk", type=int, default=1, help="top_p")
    parser.add_argument("--recent_k", type=int, default=50, help="top_p")
    parser.add_argument("--agg_temperature", type=float, default=1, help="temperature")
    parser.add_argument('--sample', default=False, action='store_true')
    parser.add_argument("--sample_topk", type=int, default=10, help="topk")
    parser.add_argument("--sample_temperature", type=float, default=1, help="top_p")
    parser.add_argument("--sample_top_p", type=float, default=None, help="top_p")
    parser.add_argument("--shared_ratio", type=float, default=1, help="shared ratio")
    
    parser.add_argument("--cluster", type=int, default=-1)
    parser.add_argument("--remain_rate", default=-1, type=float)
    return parser

task_name_dict = {
    'news_headline': 'LaMP_4',
    'abstract_generation': 'abstract_generation',
    'pwab': 'pwab_pos'
}

args = get_args_parser()
args = args.parse_args()
task_name = task_name_dict[args.task_name]
args.test_data_path = f"../pa_back/data/{task_name}/processed/seen_test.pkl"
if args.remain_rate > 0:
    args.train_data_path = f"../pa_back/data/{task_name}/processed/train_{args.remain_rate}.pkl"
else:
    args.train_data_path = f"../pa_back/data/{task_name}/processed/train.pkl"
args.train_history = f"../pa_back/data/{task_name}/processed/train_ranked.json"
if args.cluster >= 0:
    args.test_data_path = f"../pa_back/data/{task_name}/processed/seen_test_{args.cluster}.pkl"
else:
    args.test_data_path = f"../pa_back/data/{task_name}/processed/seen_test.pkl"
args.test_history = f"../pa_back/data/{task_name}/processed/seen_test_ranked.json"
# args.output_dir = f"./output/{args.task_name}/task-base_LLM"


# with open(f'./data/{args.task_name}/profile-id2text.json', 'r') as f:
#     all_profile = json.load(f)
with open(f"../pa_back/data/{task_name}/processed/seen_test_ranked.json", 'r') as f:
    ranking_dict = json.load(f)
with open('./prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)

with open(args.train_history, 'r') as f:
    train_history = json.load(f)
with open(args.train_data_path, 'rb') as f:
    raw_train_data = pickle.load(f)
import random

def process_train_data(user, k, recent_k=50):

    train_data = []
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

    # user_profile = all_profile[str(user['user_id'])]

    # for k in k_list:
    if args.task_name == 'news_headline':

        for q in user['profile'][-args.recent_k:]:
            for key, value in q.items():
                q[key] = get_first_k_tokens(extract_article(str(q[key])), 768)

            prompt = prompt_template[args.task_name]['OPPU_input'].format(**q)
            full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**q)

            train_data.append(
                {
                    "prompt": prompt,
                    "full_prompt": full_prompt
                }
            )
    elif args.task_name == "pwab":
        user = raw_train_data[user['user_id']]
        for q in user:
            q['tool_call'] = q['output']['tool_call']
            prompt = prompt_template[args.task_name]['OPPU_input'].format(**q)
            full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**q)
            ranked_his = get_his(args.task_name, str(q['id']), args.k, train_history)

            train_data.append(
                {
                    "prompt": ranked_his + '\n' + prompt,
                    "full_prompt": ranked_his + '\n' + full_prompt
                }
            )
            
    else:
        p_id = str(user['id'])
        ranked_profiles = ranking_dict[p_id]
        for q in ranked_profiles:
            for key, value in q.items():
                q[key] = get_first_k_tokens(extract_article(str(q[key])), 768)
        if 'title' in q:
            q['input'] = q['title']
            q['output'] = q['abstract']
        prompt = prompt_template[args.task_name]['OPPU_input'].format(**q)
        full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**q)

        train_data.append(
            {
                "prompt": prompt,
                "full_prompt": full_prompt
            }
        )


    return train_data


def process_profile_test_data(user, batch_size, k_list):
    out_list = []
    test_question_list = [] 
    question_id_list = []
    retrieval_test_question_list = [[] for _ in range(len(k_list))]

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

    with open('./prompt/prompt.json', 'r') as f:
        prompt_template = json.load(f)
        
    user_profile = all_profile[str(user['user_id'])]

    for q in user['query']:

        if args.task_name == 'citation':
            test_question = q['input']
            test_article = extract_citation_title(test_question)
            option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
            test_prompt = prompt_template[args.task_name]['prompt'].format(test_article, option1, option2)
        elif args.task_name == 'pwab':
            test_question = q['input']
            test_article = extract_article(test_question)
            test_prompt =  prompt_template[args.task_name]['prompt'].format(test_article)
        else:
            test_question = q['input']
            test_article = extract_article(test_question)
            test_prompt =  prompt_template[args.task_name]['prompt'].format(test_article)

        test_prompt = f'### User Profile:\n{user_profile}\n\n' + test_prompt

        test_question_list.append(test_prompt)
        question_id_list.append(q['id'])

        # test_question = q['input']
        # test_article = extract_article(test_question)
        # test_prompt = '### User Profile:\n{}\n\n### User Instruction:\nWhich tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]\nDescription: {} Tag:'.format(user_profile, test_article)
        # test_question_list.append(test_prompt)
        # question_id_list.append(q['id'])

    # elif k>0:
    visible_history_list = user['profile']
    for p in visible_history_list:
        for key, value in p.items():
            p[key] = get_first_k_tokens(p[key], 368)

    history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]

    tokenized_corpus = [doc.split(" ") for doc in history_list]
    bm25 = BM25Okapi(tokenized_corpus)

    for idx, k in enumerate(k_list):
        for q in user['query']:
            test_question = q['input']
            test_article = extract_article(test_question)

            tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_article).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=k)
        
            history_string = "".join(retrieved_history)

            test_prompt = prompt_template[args.task_name]['prompt'].format(test_article)
            test_prompt = f'### User History:\n{history_string}\n\n' + test_prompt

            test_prompt = f'### User Profile:\n{user_profile}\n\n' + test_prompt

            retrieval_test_question_list[idx].append(test_prompt)
            # question_id_list.append(q['id'])
        

    test_batch_list = split_batch(test_question_list, batch_size)
    out_list.append(test_batch_list)

    for i, k in enumerate(k_list):
        out_list.append(split_batch(retrieval_test_question_list[i], batch_size))

    all_test_question_list = [test_question_list] + retrieval_test_question_list

    return out_list, question_id_list, all_test_question_list



def process_test_data(user, ranked_dict, batch_size, k):
    out_list = []
    test_question_list = [] 
    question_id_list = []
    golds_list = []
    retrieval_test_question_list = []
    raw_list = []

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
    elif args.task_name == 'pwab':
        extract_article = lambda x: x
        format_flag = True

    with open('./prompt/prompt.json', 'r') as f:
        prompt_template = json.load(f)
        
    # for user in data:
    
    # if k==0:
    visible_history_list = []
    for q in user:

        if args.task_name == 'citation':
            test_question = q['input']
            test_article = extract_citation_title(test_question)
            option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
            test_prompt = prompt_template[args.task_name]['prompt'].format(test_article, option1, option2)
        elif args.task_name == 'pwab':
            test_article = q
        else:
            test_question = q['input']
            test_article = extract_article(test_question)
            # test_prompt =  prompt_template[args.task_name]['prompt'].format(test_article)

        # test_prompt = f'### User Profile:\n{user_profile}\n\n' + test_prompt
        ranked_his = get_his(args.task_name, str(q['id']), k, ranked_dict)
        form = 'raw' if args.task_name == 'abstract_generation' else 'python'
        test_prompt = build_rag_instruction(task_name, form, test_article, ranked_his)
        test_question_list.append(test_prompt)
        question_id_list.append(q['id'])
        golds_list.append(q['output'])
        raw_list.append(q)
        # visible_history_list += q['profile']

        # test_question = q['input']
        # test_article = extract_article(test_question)
        # test_prompt = '### User Profile:\n{}\n\n### User Instruction:\nWhich tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]\nDescription: {} Tag:'.format(user_profile, test_article)
        # test_question_list.append(test_prompt)
        # question_id_list.append(q['id'])

    # elif k>0:
    # visible_history_list = user['profile']
    # for p in visible_history_list:
    #     for key, value in p.items():
    #         p[key] = get_first_k_tokens(p[key], 368)

    # history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]

    # tokenized_corpus = [doc.split(" ") for doc in history_list]
    # bm25 = BM25Okapi(tokenized_corpus)

    # for idx, k in enumerate(k_list):
    #     for q in user['query']:
    #         test_question = q['input']
    #         test_article = extract_article(test_question)

    #         tokenized_query = prompt_template[args.task_name]['retrieval_query_wokey'].format(test_article).split(" ")
    #         retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=k)
        
    #         history_string = "".join(retrieved_history)

    #         test_prompt = prompt_template[args.task_name]['prompt'].format(test_article)
    #         test_prompt = f'### User History:\n{history_string}\n\n' + test_prompt

    #         # test_prompt = f'### User Profile:\n{user_profile}\n\n' + test_prompt

    #         retrieval_test_question_list[idx].append(test_prompt)
            # question_id_list.append(q['id'])
    

    test_batch_list = split_batch(test_question_list, batch_size)
    out_list.append(test_batch_list)

    # for i, k in enumerate(k_list):
    #     out_list.append(split_batch(retrieval_test_question_list[i], batch_size))

    all_test_question_list = [test_question_list] + retrieval_test_question_list

    return out_list, question_id_list, all_test_question_list, golds_list, raw_list



def get_all_history_id(data, tokenizer_path, max_length):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    
    # prompt_all = []
    example_all = []
    label_all = []
    
    for ann in data:
        prompt = ann['prompt']
        example = ann['full_prompt']

        prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)
        # prompt_all.append(prompt)
        example = torch.tensor(tokenizer.encode(example), dtype=torch.int64)
        example_all.append(example)

        labels = copy.deepcopy(example)

        #####################################
        labels[: len(prompt)] = -1
        #######################################
        label_all.append(labels)
    
    trimmed_examples = [example[:max_length] for example in example_all]
    trimmed_labels = [label[:max_length] for label in label_all]
    
    # Determine the maximum sequence length after trimming but capped at max_length
    max_length = min(max([len(example) for example in trimmed_examples]), max_length)

    # Pad sequences to the determined max_length
    padded_examples = torch.stack([torch.cat((example, torch.zeros(max_length - len(example), dtype=torch.int64) - 1)) if len(example) < max_length else example for example in trimmed_examples])
    padded_labels = torch.stack([torch.cat((label, torch.zeros(max_length - len(label), dtype=torch.int64) - 1)) if len(label) < max_length else label for label in trimmed_labels])

    example_masks = padded_examples.ge(0)
    label_masks = padded_labels.ge(0)

    padded_examples[~example_masks] = 0
    padded_labels[~label_masks] = 0

    # example_masks = example_masks.float()
    # label_masks = label_masks.float()

    return padded_examples, padded_labels

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


def main(args):
    pwab_res = {
        "FACC": [], 
        "RACC":{
            "search": [],
            "recommend": [],
            "review": []
        }
    }
    torch.set_default_device('cuda')

    # misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed # + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    with open(args.test_data_path, 'rb') as f:
        test_data = pickle.load(f)
        test_users = list(test_data.values())
    with open(args.test_history, 'r') as f:
        ranked_dict = json.load(f) 

    # with open(args.anchor_dir, 'r') as f:
    #     anchor_user_info = json.load(f)

    # with open(args.test_dir, 'r') as f:
    #     test_user_info = json.load(f)

    # test_users = []
    # for user in test_user_info:
    #     test_users.append(all_user_data[user['list_idx']])
    #     assert str(all_user_data[user['list_idx']]['user_id']) == str(user['user_id'])

    # with open('/afs/crc.nd.edu/group/dmsquare/vol3/ztan3/LoRA-composition/LaMP_data_final/movie/cold-start/test_users.json', 'r') as f:
    #     test_users = json.load(f)

    # define the model
    model = load_model(
        ckpt_dir=args.llama_model_path,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        # lora_path=args.lora_ckpt,
        w_lora=args.w_lora,
        grad_ckpt=args.grad_ckpt
    )

    model.to(device)
    model.merge_lora_parameters()
    print('merged!!')
    model.set_all_frozen()

    model.print_trainable_params()
    # model.get_new_lora()

    # print("Model = %s" % str(model))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)

    pred_all = [[]]
    retrieval_pred_all = [[]]
    ##################################################################################################################################
    files = os.listdir(args.gate_dir)    
    # lora_path_list = [os.path.join(args.anchor_dir, i, f'lora_ckpt_{args.k}_{args.remain_rate}.pt') for i in files]
    # gate_path_list = [os.path.join(args.gate_dir, i, f'gate_ckpt_{args.k}_{args.remain_rate}.pt') for i in files]
    lora_path_list = [os.path.join(args.anchor_dir, i, f'lora_ckpt_{args.k}.pt') for i in files]
    gate_path_list = [os.path.join(args.gate_dir, i, f'gate_ckpt_{args.k}.pt') for i in files]
    ##################################################################################################################################
    for idx, user in tqdm(enumerate(test_users), total=len(test_users)):

        # user_out_dir = os.path.join(args.output_dir, 'user_{}'.format(idx))

        # Path(user_out_dir).mkdir(parents=True, exist_ok=True)

        model.reset_lora_parameters()

        data_list = []
        raw_problems = []
        for s in user:
            raw_problems.append(s)
            data = process_train_data(s, args.k, recent_k=args.recent_k)
            for d in data:
                if d not in data_list:
                    data_list.append(d)
        # print(len(data_list))

        input_ids, labels = get_all_history_id(data_list, args.tokenizer_path, args.max_seq_len)
        print(input_ids.size())
        print(f"Start selecting")
        start_time = time.time()
        
        model.get_new_lora(
            lora_path_list=lora_path_list,
            gate_path_list=gate_path_list,
            input_ids=input_ids, 
            labels=labels,
            batch_size = args.batch_size,
            topk = args.topk,
            epoch=args.epochs,
            temperature=args.agg_temperature,
            sample=args.sample, 
            sample_topk=args.sample_topk,
            sample_temperature=args.sample_temperature,
            sample_top_p = args.sample_top_p,
            shared_ratio = args.shared_ratio
        )
        # torch.save(model.lora_state_dict(), os.path.join(user_out_dir, 'lora_ckpt.pt'))
    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Selecting time {}".format(total_time_str))

        # Inference stage
        
        generator = load_generator_from_trained(model, args.tokenizer_path)
        test_batch_list, test_id_list, test_question_list, golds_list, raw_list = process_test_data(user, ranked_dict, batch_size=args.batch_size, k=0)


        for idx, setting in enumerate(test_batch_list):
            all_results = []

            for batch in setting:
                print(Fore.GREEN + batch[0])
                results = generator.generate(batch, max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p)
                all_results += results
            assert len(raw_problems) == len(all_results)
                # print(results)

            for i in range(len(all_results)):
                output = all_results[i].replace(test_question_list[idx][i], "")
                print(output)
                # try:
                #     output = re.search(r'print\(["\'](.*?)["\']\)', output).group(1)
                # except:
                if '```json' in output:
                    output = output.split('```json')[1].split('```')[0]
                if '```python' in output:
                    output = output.split('```python')[1].split('```')[0]
                    try:
                        output_buffer = io.StringIO()
                        with contextlib.redirect_stdout(output_buffer):
                            exec(output)
                        output = output_buffer.getvalue()
                    except:
                        output = re.findall(r'"(.*?)"', output)
                        if output:
                            output = output[0]
                else:
                    try:
                        output = re.search(r'print\(["\'](.*?)["\']\)', output).group(1)
                    except:
                        pass
                if '.' in output:
                    output_ls = output.split('.')
                    ol_psd = list(set(output_ls))
                    ol_psd.sort(key=output_ls.index)
                    if ol_psd[-1] and len(output_ls) != len(ol_psd):
                        ol_psd = ol_psd[:-1]
                    output = '.'.join(ol_psd)
                print(Fore.YELLOW + output)
                pred_all[idx].append({
                    "id": test_id_list[i],
                    "generation": output,
                    "output": golds_list[i],
                    })
                if args.task_name == 'pwab':
                    output_dict = pred_all[idx][-1]
                    problem_instance = raw_list[i]
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
                    pwab_res["RACC"][raw_problems[i]['type']].append(res[1])

        # test_batch_list, test_id_list, test_question_list = process_profile_test_data(user, batch_size=args.batch_size, k_list=args.k_list)


        # for idx, setting in enumerate(test_batch_list):
        #     all_results = []

        #     for batch in setting:
        #         results = generator.generate(batch, max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p)
        #         all_results += results
        #         # print(results)

        #     for i in range(len(all_results)):
        #         output = all_results[i].replace(test_question_list[idx][i], "")
        #         retrieval_pred_all[idx].append({
        #             "id": test_id_list[i],
        #             "output": output,
        #             })


    name_list = [args.k]

    for idx, name in enumerate(name_list):
        output_file = {
            'task': args.task_name,
            'golds': pred_all[idx],
        }
        if args.sample:
            with open(os.path.join(args.output_dir, 'output-Composition-topk{}-k{}-epoch{}-aggtemp{}-sample-topk{}-temp{}-recent{}.json'.format(args.topk, name, args.epochs, args.agg_temperature, args.sample_topk, args.sample_temperature, args.recent_k)), 'w') as f:
                json.dump(output_file, f, indent=4)
        elif args.sample_top_p is not None:
            with open(os.path.join(args.output_dir, 'output-Composition-topk{}-k{}-epoch{}-aggtemp{}-topp{}-sampletemp{}-recent{}.json'.format(args.topk, name, args.epochs, args.agg_temperature, args.sample_top_p, args.sample_temperature, args.recent_k)), 'w') as f:
                json.dump(output_file, f, indent=4)
        else:
            with open(os.path.join(args.output_dir, 'output-Composition-k{}-epoch{}_{}_{}.json'.format(name, args.epochs, args.remain_rate, args.cluster)), 'w') as f:
                json.dump(output_file, f, indent=4)
    if args.task_name == 'pwab':
        evaluation_res = f'./res/{args.task_name}/k{args.k}-{args.remain_rate}'
        os.makedirs(evaluation_res, exist_ok=True)
        pwab_res["FACC"] = sum(pwab_res["FACC"]) / len(pwab_res["FACC"])
        for func, acc in pwab_res["RACC"].items():
            pwab_res["RACC"][func] = sum(acc) / max(len(acc), 1)
        with open(os.path.join(evaluation_res, 'res.json'), 'w') as f:
            json.dump(pwab_res, f, indent=4)
        print(pwab_res)
    # for idx, name in enumerate(name_list):
    #     output_file = {
    #         'task': name2taskid[args.task_name],
    #         'golds': retrieval_pred_all[idx],
    #     }
    #     if args.sample:
    #         with open(os.path.join(args.output_dir, 'output-Composition-topk{}-k{}-epoch{}-aggtemp{}-sample-topk{}-temp{}-recent{}-profile.json'.format(args.topk, name, args.epochs, args.agg_temperature, args.sample_topk, args.sample_temperature, args.recent_k)), 'w') as f:
    #             json.dump(output_file, f, indent=4)
    #     elif args.sample_top_p is not None:
    #         with open(os.path.join(args.output_dir, 'output-Composition-topk{}-k{}-epoch{}-aggtemp{}-topp{}-sampletemp{}-recent{}-profile.json'.format(args.topk, name, args.epochs, args.agg_temperature, args.sample_top_p, args.sample_temperature, args.recent_k)), 'w') as f:
    #             json.dump(output_file, f, indent=4)
    #     else:
    #         with open(os.path.join(args.output_dir, 'output-Composition-topk{}-k{}-epoch{}-aggtemp{}-greedy-recent{}-profile.json'.format(args.topk, name, args.epochs, args.agg_temperature, args.sample_top_p, args.sample_temperature, args.recent_k)), 'w') as f:
    #             json.dump(output_file, f, indent=4)

    
if __name__ == "__main__":

    # args = get_args_parser()
    # args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
