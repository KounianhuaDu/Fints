import json
import argparse
from colorama import Fore, Back, Style, init
from tqdm import tqdm
import pickle as pkl
init(autoreset=True)
import os
from tqdm import tqdm
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import numpy as np
import sys
import re
from rouge import Rouge
import time
from copy import deepcopy
import torch
from vllm import LLM, SamplingParams
import torch.nn.functional as F
import signal

from data_process import pretty_history, mini_pretty_history
from instruction import get_his, SYS_PROMPT_SINGLE, build_rag_instruction, SYS_PROMPT_POS
from pwab import functions, data

def handler(sig, frame):
    print("Exiting gracefully...")
    sys.exit(0)
signal.signal(signal.SIGINT, handler)  
    
functions_dict = {tool.__name__: tool for tool in functions}
init_data = data
func_cnt = {
    "add_product_review": [0, 0],
    "get_recommendations_by_history": [0, 0],
    "search_product_by_query": [0, 0]
}

def generate_response_api(
    args,
    tokenizer,
    prompt: str,
    top_k: int,
    max_length: int = 256,
    system_message: str = None,
    temperature: float = 0,
):
    
    sys_msg = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>/n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>/n"
    )
    # Prepare the prompt by combining system_message and user prompt
    full_prompt = (
        sys_msg
        + "\n"
        + prompt
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    
    output = llm.generate(full_prompt, sampling_params)
    message = output[0].outputs[0].text
    
    message = message.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

    return message
    
    
    
def generate(args, tokenizer, problem_instance, history=""):
    p_id = problem_instance['id']
    raw_prompt = problem_instance['input']
    
    # print(Fore.GREEN + raw_prompt)
    sys_prompt = SYS_PROMPT_POS
    output = generate_response_api(args, tokenizer, raw_prompt, top_k=1, max_length=1024, system_message=sys_prompt)
    print(Fore.YELLOW + output)
    if '</think>' in output:
        output = output.split('</think>')[-1]
    # print(output.split('\n')[0])
    #exit()
    if args.form == 'json':
        if '```json' in output:
            output = output.split('```json')[1].split('```')[0]
        try:
            output = json.loads(output)
            # output = output['headline'] if 'headline' in output else output['Headline']
        except Exception as e:
            print(e)
            output = {}
    elif args.form == 'python':
        try:
            output = re.search(r'print\(["\'](.*?)["\']\)', output).group(1)
        except Exception as e:
            print(e)
            output = ""
    output_dict = {
        'id': p_id,
        'generation': output,
        'output': problem_instance['output']
    }

    return output_dict

def process_sample(samples, user, args, calibration_data, calibration_ranked):
    results = []
    if args.dataset == 'pwab': # Positive samples
        ranked_his = get_his(args.dataset, str(samples['id']), args.k, calibration_ranked)
        pos_raw_prompt = build_rag_instruction('pwab_pos', args.form, samples, ranked_his)
        qa_line = {'id': samples['id'], 'input': pos_raw_prompt, 'output': samples['output']}
        if samples['type'] == 'review':
            action = {
                "name": "add_product_review",
                "arguments":{
                    'review': samples['output']['review']['text']
                }
            }
        else:
            output_dict = generate(args, tokenizer, qa_line)
            if not output_dict['generation']:
                return []
            action = output_dict['generation']['tool_call']
            
        if action['name'] in functions_dict:
            try:
                all_data = deepcopy(init_data)
                obs = functions_dict[action["name"]](
                    data=all_data, **action["arguments"]
                )
                pos_res = calculate_reward(samples, action['name'], obs)
                print(pos_res)
                if pos_res[0] == 1:
                    if samples['type'] == 'search' and pos_res[1] > 0.65:
                        samples['output']['tool_call'] = action
                        results.append(samples)
                    elif samples['type'] == 'recommend':
                        samples['output']['tool_call'] = action
                        results.append(samples)
                    elif samples['type'] == 'review' and pos_res[1] > 0.51:
                        samples['output']['tool_call'] = action
                        results.append(samples)
            except Exception as e:
                print(f"Error: {e}")
        print(func_cnt)

    return results

def calculate_reward(task, action, observation):
    res = [0, 0.0]
    if action in func_cnt:
        func_cnt[action][0] += 1
    if task['type'] == 'search':
        func_cnt['search_product_by_query'][1] += 1
        if action == 'search_product_by_query':
            res[0] = 1
        target_asin = task['output']['product_info']['parent_asin']
        if isinstance(observation, list):
            for i in range(len(observation)):
                if target_asin in observation[i]:
                    res[1] = 1 - i/len(observation)
                    break

    elif task['type'] == 'recommend':
        func_cnt['get_recommendations_by_history'][1] += 1
        if action == 'get_recommendations_by_history':
            res[0] = 1
        target_asin = task['output']['product_info']['parent_asin']
        if isinstance(observation, list):
            for i in range(len(observation)):
                if target_asin in observation[i]:
                    res[1] = 1 - i/len(observation)
                    break

    elif task['type'] == 'review':
        func_cnt['add_product_review'][1] += 1
        if action == 'add_product_review':
            res[0] = 1
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## dataset related
    parser.add_argument(
        "--dataset", default="LaMP_4", help="Dataset to use, default: LaMP_4"
    )
    parser.add_argument("--data_path", default="./data", help="Path to save the data")
    
    ## output & log
    parser.add_argument(
        "--modelweight",
        default="../model_weights",
        help="Path to save the model weights.",
    )
    parser.add_argument(
        "--k", type=int, default=5
    )

    parser.add_argument(
        "--llm",
        default='llama-3.1',
        type=str
    )
    parser.add_argument(
        "--form",
        default='raw',
        type=str
    )
    parser.add_argument(
        "--rag",
        type=str,
        choices=["history", "others"],
        default="others",
    )
    parser.add_argument(
        "--num_neg_per_user",
        type=int,
        default=2
    )
    
    args = parser.parse_args()
    
    
    args.base_model_addr = os.path.join("/inspire/hdd/global_user/zhangweinan-24046", 'Meta-Llama-3.1-8B-Instruct')
        
    with open(os.path.join(args.data_path, args.dataset, 'processed', 'train_ranked.json'), 'r') as f:
        calibration_ranked = json.load(f)
    with open(os.path.join(args.data_path, args.dataset, 'processed', 'train.pkl'), 'rb') as f:
        calibration_data = pkl.load(f)
        u_ids = list(calibration_data.keys())
    with open(os.path.join(args.data_path, args.dataset, 'processed', 'remain_train_ranked.json'), 'r') as f:
        remain_calibration_ranked = json.load(f)
    with open(os.path.join(args.data_path, args.dataset, 'processed', 'remain_train.pkl'), 'rb') as f:
        remain_calibration_data = pkl.load(f)
        remain_u_ids = list(remain_calibration_data.keys())
        
    llm = LLM(
        model=args.base_model_addr,              # 本地模型路径或HuggingFace名称
        tensor_parallel_size=torch.cuda.device_count(),  # 用所有GPU
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_addr,
        use_fast=False,
    )
    sampling_params = SamplingParams(
        max_tokens=1024,                     
        temperature=0.01, 
    )

    pos_neg_samples = {}
    all_samples = 0
    for idx, user in enumerate(u_ids):
        pos_neg_samples[user] = []
        quality_samples = 0

        for samples in calibration_data[user]:
            res = process_sample(samples, user, args, calibration_data, calibration_ranked)
            pos_neg_samples[user].extend(res)
            quality_samples += len(res)
            print(f"[User {user}] Have generated {quality_samples} samples")
    with open(os.path.join(args.data_path, 'pwab_pos', 'processed', 'train.pkl'), 'wb') as f:
        pkl.dump(pos_neg_samples, f)
        
    remain_samples = {}
    remain_cnt = 0
    while remain_cnt < 10000:
        user = random.choice(remain_u_ids)
        sample = random.choice(remain_calibration_data[user])
        # import pdb; pdb.set_trace()
        res = process_sample(sample, user, args, remain_calibration_data, remain_calibration_ranked)
        if res:
            if user not in remain_samples:
                remain_samples[user] = []
            remain_samples[user].extend(res)
            remain_cnt += 1
    with open(os.path.join(args.data_path, 'pwab_pos', 'processed', 'remain_train.pkl'), 'wb') as f:
        pkl.dump(remain_samples, f)
        
    
        

        
    
        
    
    