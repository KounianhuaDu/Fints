import json
import argparse
from colorama import Fore, Back, Style, init
from tqdm import tqdm
import pickle as pkl
from evaluators.eval_lamp import evaluate_task, calculate_reward
init(autoreset=True)
import os
import torch
torch.cuda.manual_seed(42)
torch.manual_seed(1)
from copy import deepcopy
import time
try:
    from pwab import functions, data

    functions_dict = {tool.__name__: tool for tool in functions}
    init_data = data
except:
    pass


def main(problems, model, k, output_path, task_name, tmp_path, steering=False):
    outs = []
    pwab_res = {
        "FACC": 0, 
        "RACC":{
            "search": [],
            "recommend": [],
            "review": []
        }
    }
    for problem_instance in tqdm(problems):
        # Generate Code & Trace
        if steering:
            res = model.generate_with_steering(problem_instance, k)
        else:
            res = model.generate(problem_instance, k)
        if res:
            output_dict = res
        else:
            print(f"Generation Error for problem {problem_instance['id']}.")
            continue
        
        with open(os.path.join(tmp_path, f"{problem_instance['id']}.json"), 'w') as f:
            json.dump(output_dict, f)
        if model.args.form == 'python':
            try:
                output = re.findall(r'print\(["\'](.*?)["\']\)', output_dict['generation'])
                output_dict['generation'] = '\n'.join(output)
            except Exception as e:
                print(e)
                output_dict['generation'] = ""
        elif model.args.form == 'json':
            try:
                output_dict['generation'] = json.loads(output_dict['generation'])
            except Exception as e:
                print(e)
                output_dict['generation'] = ""
               
        if task_name in ['pwab', 'pwab_pos']:
            try:
                action = output_dict['generation']['tool_call']
                all_data = deepcopy(init_data)
                obs = functions_dict[action["name"]](
                    data=all_data, **action["arguments"]
                )
                res = calculate_reward(problem_instance, action['name'], obs)
            except Exception as e:
                print(f"Error: {e}")
                res = [0, 0.0]
            pwab_res["FACC"] += res[0] / len(problems)
            pwab_res["RACC"][problem_instance['type']].append(res[1])
            
        outs.append(output_dict)
    if task_name in ['pwab', 'pwab_pos']:
        for func, acc in pwab_res["RACC"].items():
            pwab_res["RACC"][func] = sum(acc) / max(len(acc), 1)
        with open(evaluation_res, 'w') as f:
            json.dump(pwab_res, f, indent=4)
        print(pwab_res)
     
    # print(outs)   
    with open(os.path.join(output_path), "w") as f:
        json.dump(
            {
                "task": task_name,
                "golds": outs,
            },
            f,
            indent=4
        )   
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## dataset related
    parser.add_argument(
        "--dataset", default="LaMP_4", help="Dataset to use, default: APPS"
    )
    parser.add_argument("--data_path", default="../pa_back/data", help="Path to save the data")
    
    ## output & log
    parser.add_argument(
        "--out_path", default="../pa_back/output/generation", help="Path to save the output"
    )
    parser.add_argument(
        "--tmp_path", default="../pa_back/output/tmp", help="Path to save the output"
    )
    parser.add_argument(
        "--res_path", default="../pa_back/output/res", help="Path to save the output"
    )

    ## backbone LLM
    parser.add_argument("--arch", default="llama-3.1")
    parser.add_argument(
        "--modelweight",
        default="/inspire/hdd/global_user/zhangweinan-24046",
        help="Path to save the model weights.",
    )
    
    ## algo
    parser.add_argument(
        "--algo", default="zeroshot", help="algorithm"
    )
    parser.add_argument(
        "--k", type=int, default=5
    )
    parser.add_argument(
        "--form", type=str, default='raw', choices=['raw', 'json', 'python']
    )
    parser.add_argument("--contriever_checkpoint", default="/inspire/hdd/global_user/zhangweinan-24046/contriever")
    parser.add_argument("--weight_steer", default=False, action='store_true')
    
    ## steering params
    parser.add_argument(
        "--steering",
        action="store_true",
        default=False,
        help="If True, enable steering.",
    )
    parser.add_argument("--vector_root", type=str, default='../pa_back/caa_data/caa_vector_pt/llama-3.1_caa_python_LaMP_4_0.15_mlp')
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--multipliers", nargs="+", type=float)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--act_location", type=str, default='whole', help="Where to add the steering vector. Default is 'whole'.")
    
    ## vllm
    parser.add_argument("--vllm", action="store_true", help="If True, use vllm.")
    
    ## resume checkpoint path
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="If True, load a tuned model.",
    )
    parser.add_argument(
        "--tuned_path",
        default="../tuned_models",
        help="Root path to save the checkpoints.",
    )
    parser.add_argument(
        "--model_file",
        default="",
        help="Checkpoint name. Valid only if resume is enabled.",
    )
    parser.add_argument(
        "--check_point",
        default="",
        help="Checkpoint name. Valid only if resume is enabled.",
    )

    ## LORA related
    parser.add_argument("--lora", action="store_true")
    parser.add_argument(
        "--lora_rank", type=int, default=8, help="LoRA rank for lora/qlora"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha for lora/qlora"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout for lora/qlora"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="all",
        help="If 'default', uses peft defaults. Use 'all' for our best guess for Llama models",
    )
    parser.add_argument("--plugin", default=False, action='store_true')
    
    # Steering related
    parser.add_argument(
        "--vector_data_path",
        type=str,
        default="../pa_back/caa_data",
    )
    parser.add_argument("--data_name", type=str, default="caa_python_LaMP_4_0.15_qwen3_others")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--cluster", type=int, default=-1)
    

    # Generate or eval
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="If True, enable eval mode.",
    )

    args = parser.parse_args()

    print(args)

    # Dataset loading
    test_name = f'seen_test_{args.cluster}' if args.cluster >= 0 else 'seen_test'
    with open(os.path.join(args.data_path, args.dataset, 'processed', f'{test_name}.pkl'), 'rb') as f:
        data = pkl.load(f)
        problems = []
        for u_id, samples in data.items():
            problems += samples
    print(f"Got {len(problems)} problems.")
    with open(os.path.join(args.data_path, args.dataset, 'processed', 'seen_test_ranked.json'), 'r') as f:
        ranking_dict = json.load(f)
    
    # Path info
    vector_info = os.path.basename(args.vector_root)
    os.makedirs(os.path.join(args.out_path, f"{args.algo}_{args.k}_{args.arch}_{args.cluster}_{args.plugin}", f"{vector_info}_{args.layers[0]}_{args.form}_{args.multipliers[0]}_{args.alpha}_{args.beta}_{args.weight_steer}"), exist_ok=True)
    output_path = os.path.join(args.out_path, f"{args.algo}_{args.k}_{args.arch}_{args.cluster}_{args.plugin}", f"{vector_info}_{args.layers[0]}_{args.form}_{args.multipliers[0]}_{args.alpha}_{args.beta}_{args.weight_steer}", 'generation.json')
    os.makedirs(os.path.join(args.res_path, f"{args.algo}_{args.k}_{args.arch}_{args.cluster}_{args.plugin}", f"{vector_info}_{args.layers[0]}_{args.form}_{args.multipliers[0]}_{args.alpha}_{args.beta}_{args.weight_steer}"), exist_ok=True)
    evaluation_res = os.path.join(args.res_path, f"{args.algo}_{args.k}_{args.arch}_{args.cluster}_{args.plugin}", f"{vector_info}_{args.layers[0]}_{args.form}_{args.multipliers[0]}_{args.alpha}_{args.beta}_{args.weight_steer}", 'res.json')
    
    os.makedirs(args.tmp_path, exist_ok=True)

    if args.eval:
        # Evaluation
        results = evaluate_task(output_path)
        print(results)
        with open(evaluation_res, "w") as f:
            json.dump(results, f)
    else:
        # Model
        if args.algo == 'zeroshot':
            from models.ZeroShot import ZeroShot
            model = ZeroShot(args)
        elif args.algo == 'rag':
            from models.RAG import RAG
            model = RAG(args, ranking_dict)
        elif args.algo == 'PASteer':
            from models.PerSteer import PerSteer
            model = PerSteer(args, ranking_dict)
    
        st = time.time()
        main(problems, model, args.k, output_path, args.dataset, args.tmp_path,  args.steering)
        print("Time:", time.time() - st)
        
        
    