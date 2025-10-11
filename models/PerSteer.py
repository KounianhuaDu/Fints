from ChatModels import *
from colorama import Fore, init
init(autoreset=True)
import json
import os
import torch
from tqdm import tqdm
import sys
sys.path.append("../")
from dataloader import (
    GenerationDataset
)
from data_process import extract_after_colon, extract_from_title, retrieve_top_k_with_contriver, batchify
from transformers import AutoModel, AutoTokenizer
from instruction import get_his, build_rag_instruction, SYS_PROMPT_SINGLE

class PerSteer:
    def __init__(self, args, ranking_dict, enhanced=False):
        self.args = args
        
        print(args.arch)
        
        if args.arch == 'llama_steer':
            self.generator = LlamaWrapper(os.path.join(args.modelweight, "Meta-Llama-3.1-8B-Instruct"))
        else:
            raise NotImplementedError
        
        if args.arch in ['gpt','deepseek']:
            print('API model.')
        else:
            total_params = sum(p.numel() for p in self.generator.model.parameters())
            print(Fore.GREEN + f"#Parameters: {total_params / 1e9:.2f}B")
        
        self.ranking_dict = ranking_dict
        self.enhanced = enhanced
        
        self.dataset = GenerationDataset()
        self.vector_dataset_all = self.dataset.get_data_for_caa(
            data_path=args.vector_data_path,
            data_name=args.data_name,
            split="train"
        )
        self.current_user = None
        self.steering_dict = dict()
        self.attn_steering_dict = dict()
        self.mlp_steering_dict = dict()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever", cache_dir=args.contriever_checkpoint)
        self.contriver = AutoModel.from_pretrained("facebook/contriever", cache_dir=args.contriever_checkpoint).to("cuda:0")
        self.output_dir = os.path.join(
            args.vector_data_path, "caa_vector_pt", f"{args.arch}_{args.data_name}_{args.act_location}_persteer"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
    
    def get_vectors(self, u_id):
        res_dir = os.path.join(self.output_dir, f"{u_id}_{self.args.layers[0]}.pt")
        if self.args.act_location == 'attnmlp':
            attn_dir = res_dir.replace("attnmlp", "attn")
            self.attn_steering_dict = torch.load(attn_dir)
            mlp_dir = res_dir.replace("attnmlp", "mlp")
            self.mlp_steering_dict = torch.load(mlp_dir)
            return
        if os.path.exists(res_dir):
            self.steering_dict = torch.load(res_dir)
            return
        
        vector_dataset = self.vector_dataset_all[str(u_id)]
        pos_tokens_list, neg_tokens_list = [], []
        vectors_dict=dict()
        if not vector_dataset:
            self.steering_dict = dict()
        
        if self.args.act_location == 'whole':
            get_activations = self.generator.get_last_activations
        elif self.args.act_location == 'attn':
            self.generator.set_save_internal_decodings(True)
            get_activations = self.generator.get_attn_activations
        elif self.args.act_location == 'mlp':
            self.generator.set_save_internal_decodings(True)
            get_activations = self.generator.get_mlp_activations
        
        for i in range(len(vector_dataset)):
            ques = vector_dataset[i]["question"]
            chosen = vector_dataset[i]["chosen"]
            rejected = vector_dataset[i]["rejected"]

            if ques and chosen and rejected:
                if 'pwab' in self.args.data_name:
                    if 'pwab_pos' in self.args.data_name:
                        chosen = json.dumps(chosen)
                        rejected = json.dumps(rejected)
                    else:                      
                        ques = ""
                if self.args.system_prompt != "":
                    if ques is not None:
                        ques = self.args.system_prompt + " " + ques
                    else:
                        ques = self.args.system_prompt

                ques_tokens = self.generator.tokenizer.encode(ques, return_tensors="pt")
                pos_tokens = self.generator.tokenizer.encode(ques + chosen, return_tensors="pt")
                neg_tokens = self.generator.tokenizer.encode(ques + rejected, return_tensors="pt")
                pos_tokens_list.append(
                    {
                        "pos_tokens": pos_tokens.to(self.generator.model.device),
                        "ques_tokens_len": ques_tokens.shape[1],
                        "pos_answer_len": pos_tokens.shape[1] - ques_tokens.shape[1],
                        "ques": ques,
                    }
                )
                neg_tokens_list.append(
                    {
                        "neg_tokens": neg_tokens.to(self.generator.model.device),
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
            pos_activations = dict([(layer, []) for layer in self.args.layers])
            neg_activations = dict([(layer, []) for layer in self.args.layers])
            self.steering_dict[p_tokens_dict['ques']] = dict()
            
            # Get positive logits
            self.generator.reset_all()
            self.generator.get_logits(p_tokens)

            for layer in self.args.layers:
                p_activations = get_activations(layer)
                p_activations = p_activations[0, ques_tokens_len:, :].mean(0).detach()
                pos_activations[layer] = p_activations.cpu()

            # Get negative logits
            self.generator.reset_all()
            self.generator.get_logits(n_tokens)

            for layer in self.args.layers:
                n_activations = get_activations(layer)
                n_activations = n_activations[0, ques_tokens_len:, :].mean(0).detach()
                neg_activations[layer] = n_activations.cpu()
            

            for layer in self.args.layers:
                self.steering_dict[p_tokens_dict['ques']][layer] = pos_activations[layer] - neg_activations[layer]
        torch.save(self.steering_dict, res_dir)
    
    def get_his(self, p_id, k):
        ranked_profiles = self.ranking_dict[p_id][:k]
        
        q_a_history = []
        for idx, sample in enumerate(ranked_profiles):
            if self.args.dataset == "LaMP_4":
                line = f"Historical sample {idx}:\n Q: {sample['text']}. \n A: {sample['title']}."
            elif self.args.dataset == "LaMP_5":
                line = f"Historical sample {idx}:\n Q: {sample['abstract']}. \n A: {sample['title']}."
            else:
                raise NotImplementedError
            q_a_history.append(line)
        q_a_history = '\n'.join(q_a_history)
        return q_a_history
    
    def generate_with_steering(self, problem_instance, k):
        # Setup model
        self.generator.reset_all()
        user_id = problem_instance['user_id']
        profile = self.vector_dataset_all[str(user_id)]
        if user_id != self.current_user:
            self.get_vectors(str(user_id))
            self.current_user = user_id
            self.corpus = [f"{x['question']} {x['chosen']}" for x in profile]
        
        if 'LaMP' in self.args.dataset:
            inp = extract_after_colon(problem_instance['input'])
        elif self.args.dataset == 'abstract_generation':
            inp = extract_from_title(problem_instance['input'])
        elif self.args.dataset in ['pwab', 'pwab_pos']:
            inp = problem_instance['input']
        if self.args.weight_steer:
            ranked_profiles, weights = retrieve_top_k_with_contriver(self.contriver, self.tokenizer, self.corpus, 
                                                        profile, inp, min(len(profile), k), return_weight=True)
            # print(weights)
        else:
            ranked_profiles = retrieve_top_k_with_contriver(self.contriver, self.tokenizer, self.corpus, 
                                                        profile, inp, min(len(profile), k))
        
        steering_vector_dict = dict([(layer, []) for layer in self.args.layers])
        attn_vector_dict = dict([(layer, []) for layer in self.args.layers])
        mlp_vector_dict = dict([(layer, []) for layer in self.args.layers])
        for p in ranked_profiles:
            for layer in self.args.layers:
                if self.args.act_location == 'attnmlp':
                    attn_vector_dict[layer].append(self.attn_steering_dict[p['question']][layer])
                    mlp_vector_dict[layer].append(self.mlp_steering_dict[p['question']][layer])
                else:
                    steering_vector_dict[layer].append(self.steering_dict[p['question']][layer])
            
        for layer, coefficient in zip(self.args.layers, self.args.multipliers):
            if steering_vector_dict[layer]:
                if self.args.weight_steer:
                    steering_vector = (torch.stack(steering_vector_dict[layer]) * weights.unsqueeze(1)).sum(dim=0).to(self.generator.device)
                else:
                    steering_vector = torch.stack(steering_vector_dict[layer]).mean(0).to(self.generator.device)
            else:
                steering_vector = torch.zeros(4096).to(self.generator.device)
            if attn_vector_dict[layer]:
                if self.args.weight_steer:
                    attn_vector = (torch.stack(attn_vector_dict[layer]) * weights.unsqueeze(1)).sum(dim=0).to(self.generator.device)
                else:
                    attn_vector = torch.stack(attn_vector_dict[layer]).mean(0).to(self.generator.device)
            else:
                attn_vector = torch.zeros(4096).to(self.generator.device)
            if mlp_vector_dict[layer]:
                if self.args.weight_steer:
                    mlp_vector = (torch.stack(mlp_vector_dict[layer]) * weights.unsqueeze(1)).sum(dim=0).to(self.generator.device)
                else:
                    mlp_vector = torch.stack(mlp_vector_dict[layer]).mean(0).to(self.generator.device)
            else:
                mlp_vector = torch.zeros(4096).to(self.generator.device)
                
            if self.args.act_location == 'whole':
                self.generator.set_add_activations(
                    layer, coefficient * steering_vector
                )
            elif self.args.act_location == 'attn':
                self.generator.set_add_attention_activations(
                    layer, coefficient * steering_vector
                )
            elif self.args.act_location == 'mlp':
                self.generator.set_add_mlp_activations(
                    layer, coefficient * steering_vector
                )
            elif self.args.act_location == 'attnmlp':
                self.generator.set_add_attention_activations(
                    layer, coefficient * attn_vector
                )
                self.generator.set_add_mlp_activations(
                    layer, coefficient * mlp_vector
                )
        
        # Prompt preparation
        k_dict = {'LaMP_4': 5, 'abstract_generation': 3, 'pwab': 5, 'pwab_pos': 5}
        p_id = str(problem_instance['id'])
        ranked_his = get_his(self.args.dataset, p_id, k_dict[self.args.dataset], self.ranking_dict)
        if self.args.dataset in ['pwab', 'pwab_pos']:
            raw_prompt = build_rag_instruction(self.args.dataset, 'raw', problem_instance, ranked_his)
        else:
            raw_prompt = build_rag_instruction(self.args.dataset, 'raw', problem_instance['input'], ranked_his)
        print(Fore.GREEN + raw_prompt)
        sys_prompt = SYS_PROMPT_SINGLE if self.args.dataset in ['pwab', 'pwab_pos'] else ""
        sys_msg = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>/n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>/n"
        )

        # Prepare the prompt by combining system_message and user prompt
        full_prompt = (
            sys_msg
            + "\n"
            + raw_prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        
        # Generate
        task_lora = None
        if self.args.plugin:
            task_name_dict = {'LaMP_4': 'news_headline', 'abstract_generation': 'abstract_generation', 'pwab_pos': 'pwab'}
            lora_k = 5 if self.args.dataset == 'pwab_pos' else 0
            task_name = task_name_dict[self.args.dataset]
            task_lora = f"./ckpt/{task_name}/k{lora_k}-{user_id}-Meta-Llama-3.1-8B-Instruct-OPPU_LoRA"
        output = self.generator.generate(
            full_prompt,
            max_new_tokens=1024,
            task_lora=task_lora
        )
        output = output.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
        print(Fore.YELLOW + output)
        output_dict = {
            'id': p_id,
            'generation': output,
            'output': problem_instance['output']
        }
        
        return output_dict
        
