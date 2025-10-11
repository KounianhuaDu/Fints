# -*- coding:utf-8 _*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import json
from .instruction import *
from colorama import Fore, init
import os
from peft import PeftModel
from vllm import LLM, SamplingParams
import pprint
from safetensors.torch import load_file

from colorama import Fore, Back, Style, init


class LlamaChat:
    def __init__(self, model_name, args):
        use_vllm = args.vllm
        self.name = model_name
        self.args = args
        if model_name == "llama3-8b":
            self.model_path = os.path.join(args.modelweight, "Meta-Llama-3.1-8B-Instruct")
        elif model_name == "llama3-3b":
            self.model_path = os.path.join(args.modelweight, "Llama-3.2-3B-Instruct")

        if use_vllm:
            self.model = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=args.device_num,
            )
            self.tokenizer = self.model.get_tokenizer()
            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=False, padding_side="left"
            )
            self.tokenizer.pad_token_id = 0  # unk. to be different from the eos token

            if args.lora:
                self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map="auto",
                # load_in_8bit=True
                # quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"),
            )
                print(Fore.RED + "RESUME")
                print(Fore.GREEN + args.check_point)
                if args.lora:
                    from peft import LoraConfig, TaskType, get_peft_model

                    if args.lora_target_modules == "all":
                        lora_target_modules = [
                            "q_proj",
                            "v_proj"
                        ]
                    elif args.lora_target_modules.lower() == "default":
                        lora_target_modules = None
                    else:
                        lora_target_modules = args.lora_target_modules.split(",")

                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=args.lora_rank,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        target_modules=lora_target_modules,
                    )
                    # self.model = get_peft_model(self.model, peft_config)
                    self.model = PeftModel.from_pretrained(self.model, args.check_point)
                    self.model = self.model.merge_and_unload()
                    print(Fore.RED + "Lora model set and merged.")
                else:
                    self.model.load_state_dict(load_file(args.check_point), strict=False)
                    print(Fore.RED + "Model dict loaded.")
                
                time.sleep(3)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map="auto",
            )

    def generate_response_api(
        self,
        prompt: str,
        top_k: int,
        max_length: int = 1024,
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
        #print(Fore.GREEN + full_prompt)

        if self.args.vllm:
            # Generate the response
            output = self.model.generate(
                full_prompt,
                # vllm get logits/ log_probs
                sampling_params=SamplingParams(
                    temperature=temperature,
                    top_k=top_k,
                    max_tokens=max_length,
                ),
            )
            log_probs_for_generated_tokens = (
                None  # Initialize to handle cases where it's not needed
            )
            message = output[0].outputs[0].text
        else:
            model_inputs = self.tokenizer([full_prompt], return_tensors="pt").to(
                self.model.device
            )
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
            attention_mask = torch.ones(
                input_ids.shape, dtype=torch.long, device=self.model.device
            )
            # Generate the response
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                pad_token_id=self.tokenizer.eos_token_id,  # Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            # Decode the response
            message = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return message
    