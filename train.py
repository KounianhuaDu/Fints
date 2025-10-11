import sys

sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

import torch
from tqdm import tqdm
import os
from ChatModels import (
    LlamaWrapper,
)
import argparse
from dataloader import (
    GenerationDataset, 
)
import json

from instruction import SYS_PROMPT_SINGLE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument(
        "--data_path",
        type=str,
        default="../pa_back/caa_data",
    )
    parser.add_argument("--data_name", type=str, default="caa_python_LaMP_4_0.15_qwen_others")
    parser.add_argument("--model_name", type=str, default="llama-3.1")
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="../model_weights/fix/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--rerun", action="store_true", default=False, help="Rerun even if files already exist")
    parser.add_argument("--act_location", type=str, choices=['whole', 'attn', 'mlp'], default='whole')

    args = parser.parse_args()
    print(args)
    if "llama" in args.model_name.lower():
        model = (
            LlamaWrapper(args.model_name_or_path)
        )
    elif "qwen" in args.model_name.lower():
        model = (
            QwenWrapper(args.model_name_or_path)
        )
    else:
        raise NotImplementedError(f"Model {args.model_name} not supported")
    tokenizer = model.tokenizer
    pos_activations = dict([(layer, []) for layer in args.layers])
    neg_activations = dict([(layer, []) for layer in args.layers])
    if args.act_location == 'whole':
        get_activations = model.get_last_activations
    elif args.act_location == 'attn':
        model.set_save_internal_decodings(True)
        get_activations = model.get_attn_activations
    elif args.act_location == 'mlp':
        model.set_save_internal_decodings(True)
        get_activations = model.get_mlp_activations

    dataset = GenerationDataset()

    vector_dataset_all = dataset.get_data_for_caa(
        data_path=args.data_path,
        data_name=args.data_name,
        split="train",
    )
    device = model.device
    if args.model_name == "gemma-2-9b" or args.model_name=="llama-3.1":
        output_dir = os.path.join(
            args.data_path, "caa_vector_pt", f"{args.model_name}_{args.data_name}_{args.act_location}"
        )
    else:
        output_dir = os.path.join(
            args.data_path, args.data_name, "caa_vector", f"{args.model_name}_{args.mode}"
        )
    for i, (uid, vector_dataset) in enumerate(vector_dataset_all.items()):
        print(i)
        pos_tokens_list, neg_tokens_list = [], []
        if os.path.exists(os.path.join(output_dir, f"{uid}_{args.layers[0]}.pt")) and not args.rerun:
            continue
        for i in range(len(vector_dataset)):
            ques = vector_dataset[i]["question"]
            chosen = vector_dataset[i]["chosen"]
            rejected = vector_dataset[i]["rejected"]

            if ques and chosen and rejected:
                if 'pwab' in args.data_name:
                    if 'pwab_pos' in args.data_name:
                        chosen = json.dumps(chosen)
                        rejected = json.dumps(rejected)
                    else:                      
                        ques = ""
                    
                if args.model_name=="llama-3.1":
                    if args.system_prompt != "":
                        if ques is not None:
                            ques = args.system_prompt + " " + ques
                        else:
                            ques = args.system_prompt
                else:
                    raise NotImplementedError


                ques_tokens = tokenizer.encode(ques, return_tensors="pt")
                pos_tokens = tokenizer.encode(ques + chosen, return_tensors="pt")
                neg_tokens = tokenizer.encode(ques + rejected, return_tensors="pt")
                pos_tokens_list.append(
                    {
                        "pos_tokens": pos_tokens.to(device),
                        "ques_tokens_len": ques_tokens.shape[1],
                        "pos_answer_len": pos_tokens.shape[1] - ques_tokens.shape[1],
                    }
                )
                neg_tokens_list.append(
                    {
                        "neg_tokens": neg_tokens.to(device),
                        "ques_tokens_len": ques_tokens.shape[1],
                        "neg_answer_len": neg_tokens.shape[1] - ques_tokens.shape[1],
                    }
                )

        for p_tokens_dict, n_tokens_dict in tqdm(
            zip(pos_tokens_list, neg_tokens_list),
            total=len(pos_tokens_list),
            desc="Processing prompts",
        ):
            p_tokens = p_tokens_dict["pos_tokens"]
            n_tokens = n_tokens_dict["neg_tokens"]
            ques_tokens_len = p_tokens_dict["ques_tokens_len"]
            
            # Get positive logits
            model.reset_all()
            model.get_logits(p_tokens)

            for layer in args.layers:
                p_activations = get_activations(layer)
                p_activations = p_activations[0, ques_tokens_len:, :].mean(0).detach()
                pos_activations[layer].append(p_activations.cpu())

            # Get negative logits
            model.reset_all()
            model.get_logits(n_tokens)

            for layer in args.layers:
                n_activations = get_activations(layer)
                n_activations = n_activations[0, ques_tokens_len:, :].mean(0).detach()
                neg_activations[layer].append(n_activations.cpu())

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for layer in args.layers:
            if (len(pos_activations[layer]) == 0):
                p_activations = model.get_last_activations(layer)
                vec = torch.zeros(4096)
            else:
                all_pos_layer = torch.stack(pos_activations[layer])
                all_neg_layer = torch.stack(neg_activations[layer])
                vec = (all_pos_layer - all_neg_layer).mean(dim=0).to(model.device) 

            torch.save(
                vec.cpu(),
                os.path.join(output_dir, f"{uid}_{layer}.pt"),
            )