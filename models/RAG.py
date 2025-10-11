from ChatModels import *
from colorama import Fore, init
init(autoreset=True)
import json
import os
import torch
from instruction import build_rag_instruction, SYS_PROMPT_SINGLE, get_his

class RAG:
    def __init__(self, args, ranking_dict, enhanced=False):
        self.args = args
        
        print(args.arch)
        if args.arch in ["llama3-8b", "llama3-3b"]:
            self.generator = LlamaChat(args.arch, args)
        elif args.arch == "deepseek":
            self.generator = DeepSeekChat(args.arch, args)
        elif args.arch == "gemma":
            self.generator = GemmaChat(args.arch, args)
        elif args.arch == "gpt":
            self.generator = GPTChat(args.arch, args)
        elif args.arch == "qwen":
            self.generator = QwenChat(args.arch, args)
        elif args.arch == "claude":
            self.generator = ClaudeChat(args.arch, args)
        elif args.arch == 'llama_steer':
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
        self.system_message = SYS_PROMPT_SINGLE if self.args.dataset in ['pwab', 'pwab_pos'] else ""

    
    def get_his(self, p_id, k):
        ranked_profiles = self.ranking_dict[p_id][:k]
        
        q_a_history = []
        for idx, sample in enumerate(ranked_profiles):
            if self.args.dataset == "LaMP_4":
                line = f"Historical sample {idx}:\n Q: {sample['text']}. \n A: {sample['title']}."
            elif self.args.dataset == "LaMP_5":
                line = f"Historical sample {idx}:\n Q: {sample['abstract']}. \n A: {sample['title']}."
            elif self.args.dataset == "abstract_generation":
                line = f"Historical sample {idx}:\n Q: {sample['title']}. \n A: {sample['abstract']}."
            else:
                raise NotImplementedError
            q_a_history.append(line)
        q_a_history = '\n'.join(q_a_history)
        return q_a_history
    
    def build_enhanced_instruction(self, prompt, his):
        if self.args.dataset == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.args.dataset == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.args.dataset == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.args.dataset == "LaMP_4":  
            inp = f"Generate a headline for the following article: {prompt}"
            inp += f"For your reference, here are the user's past QA pairs:\n {his}"
            inp += "Please generate 3 suitable headlines and wrap them in the below format. \n"
            inp += """
```json
[
{"Headline 1": "Are You Ready to Open Up to Trust, Happiness and Joy?"},
{"Headline 2": "Fear or Freedom? Choose Trust, Happiness, Joy"},
{"Headline 3": "The Journey to Trust, Happiness, and Joy Starts With a ‘Yes’"}
]
```
"""
        elif self.args.dataset == "LaMP_5":
            inp = f"Generate a title for the following abstract of a paper: {prompt}"
        elif self.args.dataset == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        return inp

    
    def generate(self, problem_instance, k):
        p_id = str(problem_instance['id'])
        
        ranked_his = get_his(self.args.dataset, p_id, k, self.ranking_dict)
        if not self.enhanced:
            if self.args.dataset in ['pwab', 'pwab_pos']:
                raw_prompt = build_rag_instruction(self.args.dataset, self.args.form, problem_instance, ranked_his)
            else:
                raw_prompt = build_rag_instruction(self.args.dataset, self.args.form, problem_instance['input'], ranked_his)
            print(Fore.GREEN + raw_prompt)
            output = self.generator.generate_response_api(raw_prompt, top_k=1, system_message=self.system_message)
            print(Fore.YELLOW + output)
            output_dict = {
                'id': p_id,
                'generation': output,
                'output': problem_instance['output']
            }
        else:
            raw_prompt = self.build_enhanced_instruction(problem_instance['input'], ranked_his)
            output = self.generator.generate_response_api(raw_prompt, top_k=1)
            print(Fore.YELLOW + output)
            
            output_dict = {
                'id': p_id,
                'user_id': problem_instance['user_id'],
                'input': problem_instance['input'],
                'generation': output,
                'output': problem_instance['output']
            }
        
        return output_dict
    
    def generate_with_steering(self, problem_instance, k):
        # Setup model
        self.generator.reset_all()
        user_id = problem_instance['user_id']
        for layer, coefficient in zip(self.args.layers, self.args.multipliers):
            steering_vector_path = os.path.join(self.args.vector_root, f"{user_id}_{layer}.pt")
            if self.args.act_location == "attnmlp":
                attn_steering_path = steering_vector_path.replace("attnmlp", "attn")
                attn_vector = torch.load(attn_steering_path).to(self.generator.device)
                self.generator.set_add_attention_activations(layer, self.args.alpha * attn_vector)
                mlp_steering_path = steering_vector_path.replace("attnmlp", "mlp")
                mlp_vector = torch.load(mlp_steering_path).to(self.generator.device)
                self.generator.set_add_mlp_activations(layer, self.args.beta * mlp_vector)
                continue
            
            steering_vector = torch.load(steering_vector_path).to(self.generator.device)
            if self.args.act_location == 'whole':
                self.generator.set_add_activations(
                    layer, coefficient * steering_vector
                )
            if self.args.act_location == 'attn':
                self.generator.set_add_attention_activations(
                    layer, coefficient * steering_vector
                )
            if self.args.act_location == 'mlp':
                self.generator.set_add_mlp_activations(
                    layer, coefficient * steering_vector
                )
        
        # Prompt preparation
        p_id = str(problem_instance['id'])
        ranked_his = get_his(self.args.dataset, p_id, k, self.ranking_dict)
        if self.args.dataset in ['pwab', 'pwab_pos']:
            raw_prompt = build_rag_instruction(self.args.dataset, self.args.form, problem_instance, ranked_his)
        else:
            raw_prompt = build_rag_instruction(self.args.dataset, self.args.form, problem_instance['input'], ranked_his)
        print(Fore.GREEN + raw_prompt)
        sys_msg = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>/n{self.system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>/n"
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
        

        
