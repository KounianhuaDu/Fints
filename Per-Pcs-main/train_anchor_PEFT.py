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
# from torch.utils.tensorboard import SummaryWriter
# from util.misc import NativeScalerWithGradNormCount as NativeScaler
from utils import split_batch, get_first_k_tokens, print_trainable_parameters, name2taskid
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, extract_news_headline, extract_product_review, extract_scholarly_title, extract_tweet_paraphrasing, extract_abstract_generation

from llama import Tokenizer
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
from tqdm import trange, tqdm
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import Fore, init
init(autoreset=True)
import ast

from instruction import get_his, build_rag_instruction
# from lora_composition import process_test_data

def extract_article(text):
    marker = "] description: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

class InstructionDataset(Dataset):
    def __init__(self, data_list, tokenizer_path, max_tokens=2048):
        self.ann = data_list

        self.max_words = max_tokens
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

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
        default=6,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--max_step", default=100, type=int)

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

    parser.add_argument("--task_name", default="news_headline", type=str, metavar="MODEL", help="name of the task")

    parser.add_argument("--model", default="llama7B_lora", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--max_seq_len", type=int, default=3000, metavar="LENGTH", help="the maximum sequence length")
    
    parser.add_argument("--w_lora", type=bool, default=True, help="use lora or not")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (absolute lr)")
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


    # Dataset parameters
    # parser.add_argument("--test_data_path", default="./data/movie_tagging/user_anchor_candidate.json", type=str, help="dataset path")
    # parser.add_argument("--train_data_path", default="/afs/crc.nd.edu/user/z/ztan3/Private/LoRA-composition/LaMP_data-final/movie/user_base_LLM.json", type=str, help="dataset path")
    
    parser.add_argument("--output_dir", default="./output/news_headline/Anchor_PEFT/LoRA", help="path where to save, empty for no saving")

    parser.add_argument("--log_dir", default="./output", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lora_ckpt", default='./output/news_headline/task-base_LLM/lora_ckpt.pt', help="resume lora from checkpoint")
    parser.add_argument("--grad_ckpt", type=bool, default=True, help="whether to user gradient checkpoint, recommend TRUE!!")

    parser.add_argument("--anchor_path", default='./anchor_selection/abstract_generation/anchor_user_idx.pt', help="resume lora from checkpoint")

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
    parser.add_argument("--max_gen_len", type=int, default=512, help="top_p")

    parser.add_argument("--k", type=int, default=3, help="top_p")
    parser.add_argument('--infer', default=False, action='store_true')
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
# args.test_data_path = f"./data/{args.task_name}/test_100/user_test_100.json"
args.train_data_path = f"./data/{args.task_name}/user_base_LLM.json"
if args.remain_rate == -1:
    args.test_data_path = f"../pa_back/data/{task_name}/processed/train.pkl"
else:
    args.test_data_path = f"../pa_back/data/{task_name}/processed/train_{args.remain_rate}.pkl"
args.test_history = f"../pa_back/data/{task_name}/processed/train_ranked.json"

# with open(f'./data/{args.task_name}/profile-id2text.json', 'r') as f:
#     all_profile = json.load(f)
with open(args.test_history, 'r') as f:
    ranked_profile = json.load(f)
    
with open(f"../pa_back/data/{task_name}/processed/seen_test.pkl", 'rb') as f:
    test_data = pickle.load(f)
with open(f"../pa_back/data/{task_name}/processed/seen_test_ranked.json", 'r') as f:
    ranking_dict = json.load(f)

import random

def process_train_data(user, k):

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
    
    with open('./prompt/prompt.json', 'r') as f:
        prompt_template = json.load(f)

    # user_profile = all_profile[str(user['user_id'])]
    if args.task_name == 'news_headline':
        
        for idx, q in enumerate(user['profile']):

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
    elif args.task_name == 'pwab':
        user['tool_call'] = user['output']['tool_call']  
        prompt = prompt_template[args.task_name]['OPPU_input'].format(**user)
        full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**user)
        if k > 0:
            history_string = get_his(args.task_name, str(user['id']), k, ranked_profile)
            prompt = history_string + "\n" + prompt
            full_prompt = history_string + "\n" + full_prompt

        import pdb; pdb.set_trace()
        train_data.append(
            {
                "prompt": prompt,
                "full_prompt": full_prompt
            }
        )
        # import pdb; pdb.set_trace()
    else:
        for key, value in user.items():
            user[key] = get_first_k_tokens(extract_article(str(user[key])), 768)
        prompt = prompt_template[args.task_name]['OPPU_input'].format(**user)
        full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**user)
        if k > 0:
            history_string = get_his(args.task_name, str(user['id']), k, ranked_profile)
            prompt = history_string + "\n" + prompt
            full_prompt = history_string + "\n" + full_prompt


        train_data.append(
            {
                "prompt": prompt,
                "full_prompt": full_prompt
            }
        )

    return train_data

def process_test_data(user, ranked_dict, batch_size, k):
    out_list = []
    test_question_list = [] 
    question_id_list = []
    golds_list = []
    retrieval_test_question_list = []

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
        
    # for user in data:
    
    # if k==0:
    visible_history_list = []
    for q in user:

        if args.task_name == 'citation':
            test_question = q['input']
            test_article = extract_citation_title(test_question)
            option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
            test_prompt = prompt_template[args.task_name]['prompt'].format(test_article, option1, option2)
        else:
            test_question = q['input']
            test_article = extract_article(test_question)
            test_prompt =  prompt_template[args.task_name]['prompt'].format(test_article)

        # test_prompt = f'### User Profile:\n{user_profile}\n\n' + test_prompt
        if k > 0:
            ranked_his = get_his(task_name, str(q['id']), k, ranked_dict)
            test_prompt = build_rag_instruction(task_name, 'raw', test_prompt, ranked_his)
        test_question_list.append(test_prompt)
        question_id_list.append(q['id'])
        golds_list.append(q['output'])
        # visible_history_list += q['profile']

    test_batch_list = split_batch(test_question_list, batch_size)
    out_list.append(test_batch_list)

    # for i, k in enumerate(k_list):
    #     out_list.append(split_batch(retrieval_test_question_list[i], batch_size))

    all_test_question_list = [test_question_list] + retrieval_test_question_list

    return out_list, question_id_list, all_test_question_list, golds_list

def main(args):
    torch.set_default_device('cuda')

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed # + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    anchor_idx = torch.load(args.anchor_path, weights_only=False)

    with open(args.test_data_path, 'rb') as f:
        all_data = pickle.load(f)
        all_user_data = list(all_data.values())
        all_uid = list(all_data.keys())
    
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
    model.print_trainable_params()
    model.merge_lora_parameters()
    print('merged!!')

    # print("Model = %s" % str(model))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)

    pred_all = [[]]

    for idx in tqdm(range(len(anchor_idx))):
        idx_all_test = anchor_idx[idx]
        uid = all_uid[idx]
        print(idx_all_test)
        # print(type(all_user_data))
        user = all_user_data[idx_all_test]

        user_out_dir = os.path.join(args.output_dir, 'user_{}'.format(user[0]['user_id']))

        Path(user_out_dir).mkdir(parents=True, exist_ok=True)

        model.reset_lora_parameters()
        model.set_lora_trainable()
        
        data_list = []
        for s in user:
            data = process_train_data(s, args.k)
            data_list += data

        dataset_train = InstructionDataset(
            data_list=data_list, tokenizer_path=args.tokenizer_path, max_tokens=args.max_seq_len
        )
    
        # sampler_train = torch.utils.data.RandomSampler(dataset_train)

        # os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.log_dir)
        # else:
        log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            # sampler=sampler_train,
            shuffle=True,
            batch_size=args.batch_size,
            # num_workers=args.num_workers,
            # pin_memory=args.pin_mem,
            drop_last=False,
            generator=torch.Generator(device='cuda'),
            collate_fn=partial(collate_fn, max_length=args.max_seq_len),
        )

        # following timm: set wd as 0 for bias and norm layers
        # param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        # loss_scaler = NativeScaler()


        args.cur_step = 0
        try:
            print(f"Start training for {args.max_step} steps")
        except:
            print(f"Start training for {args.epochs} epochs")        
        
        start_time = time.time()
        # for epoch in range(args.epochs):
        epoch = 0

        while args.cur_step < args.max_step:
            train_stats = train_one_epoch(
                model, data_loader_train, optimizer, device, epoch, None, log_writer=log_writer, args=args
            )
            epoch += 1

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                # **{f"val_{k}": v for k, v in val_stats.items()},
            }

            if args.output_dir:
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(user_out_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        torch.save(model.lora_state_dict(), os.path.join(user_out_dir, f'lora_ckpt_{args.k}_{args.remain_rate}.pt'))
    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
        
        # Inference stage
    #     model.eval()
    #     model.set_all_frozen()
    #     model.print_trainable_params()
    #     generator = load_generator_from_trained(model, args.tokenizer_path)
    #     test_user = test_data[uid]
    #     test_batch_list, test_id_list, test_question_list, golds_list = process_test_data(test_user, ranking_dict, batch_size=args.batch_size, k=args.k)


    #     for idx, setting in enumerate(test_batch_list):
    #         all_results = []

    #         for batch in setting:
    #             print(Fore.GREEN + batch[0])
    #             results = generator.generate(batch, max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p)
    #             all_results += results
    #             # print(results)
    #             # import pdb; pdb.set_trace()

    #         for i in range(len(all_results)):
    #             # print(Fore.YELLOW + all_results[i])
    #             output = all_results[i].replace(test_question_list[idx][i], "")
    #             output_ls = output.split('.')
    #             ol_psd = list(set(output_ls))
    #             ol_psd.sort(key=output_ls.index)
    #             if ol_psd[-1]:
    #                 ol_psd = ol_psd[:-1]
    #             output = '.'.join(ol_psd)
    #             print(Fore.YELLOW + output)
    #             pred_all[idx].append({
    #                 "id": test_id_list[i],
    #                 "generation": output,
    #                 "output": golds_list[i],
    #                 })
    
    # output_file = {
    #     'task': args.task_name,
    #     'golds': pred_all[0],
    # }            
    # with open(os.path.join(args.output_dir, 'output-Composition-k{}-epoch{}.json'.format(3, args.epochs), 'w')) as f:
    #     json.dump(output_file, f, indent=4)

    
if __name__ == "__main__":

    # args = get_args_parser()
    # args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
