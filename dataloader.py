import os, json, sys
from typing import Dict, Optional
from datasets import Dataset, load_dataset
from tqdm import tqdm
import numpy as np
import random
import re
import pdb
import json
import pickle

class BaseDataset:
    def __init__(self):
        pass

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.data.items()}

    def __len__(self):
        return len(self.data["input_ids"])

    def get_data(self):
        pass

    def get_data_for_sft_training(self):
        raise NotImplementedError("Not implemented")

    def get_data_for_dpo_training(self):
        raise NotImplementedError("Not implemented")

    def get_data_for_icl(self):
        raise NotImplementedError("Not implemented")

    def get_data_for_caa(self):
        raise NotImplementedError("Not implemented")

    def get_data_for_selection(self):
        raise NotImplementedError("Not implemented")

# personality generaton dataset
class GenerationDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def get_data(self):
        return self.data

    def get_few_shot_system_prompt(
        self,
        question,
        n_shots=0,
        demo_type="safe",
    ):
        if n_shots > 0:
            global PROMPT_EXAMPLARS
            if len(PROMPT_EXAMPLARS) > n_shots:
                prompt_examples = PROMPT_EXAMPLARS[:n_shots]
                print(f"n_shots: {n_shots}\n EXAMPLARS: {prompt_examples}")
            else:
                prompt_examples = PROMPT_EXAMPLARS[:]
                print(f"n_shots: {len(prompt_examples)}\n EXAMPLARS: {prompt_examples}")
            demonstrations = []
            for example in prompt_examples:
                if demo_type=="safe":
                    demonstrations.append(
                        # "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                        "Quesion: " + example["question"] + "\nAnswer: " + example["safe_answer"]
                    )
                else:
                    demonstrations.append(
                        # "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                        "Quesion: " + example["question"] + "\nAnswer: " + example["unsafe_answer"]
                    )
            # prompt_prefix = "".join(demonstrations)
            if demo_type=="safe":
                prompt_prefix = "Here are some examples of safety:\n\n" + "\n\n".join(demonstrations) + "\n\n"
            else:
                prompt_prefix = "Here are some examples:\n\n" + "\n\n".join(demonstrations) + "\n\n"
            question = prompt_prefix + "Question: " + question.strip() + "\nAnswer: "
        return question
    
    def get_few_shot_system_prompt_data_num(
        self,
        question,
        n_shots=0,
        demo_type="safe",
    ):
        if n_shots > 0:
            global PROMPT_EXAMPLARS_data_num
            if len(PROMPT_EXAMPLARS_data_num) > n_shots:
                PROMPT_EXAMPLARS_data_num = PROMPT_EXAMPLARS_data_num[:n_shots]
                print(f"n_shots: {n_shots}\n EXAMPLARS: {PROMPT_EXAMPLARS_data_num}")
            demonstrations = []
            for example in PROMPT_EXAMPLARS_data_num:
                if demo_type=="safe":
                    demonstrations.append(
                        # "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                        "Quesion: " + example["question"] + "\nAnswer: " + example["safe_answer"]
                    )
                else:
                    demonstrations.append(
                        # "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                        "Quesion: " + example["question"] + "\nAnswer: " + example["unsafe_answer"]
                    )
            # prompt_prefix = "".join(demonstrations)
            if demo_type=="safe":
                prompt_prefix = "Here are some examples of safety:\n\n" + "\n\n".join(demonstrations) + "\n\n"
            else:
                prompt_prefix = "Here are some examples:\n\n" + "\n\n".join(demonstrations) + "\n\n"
            question = prompt_prefix + "Question: " + question.strip() + "\nAnswer: "
        return question

    def get_data_for_selection(
        self,
        uid,
        data_file="power-seeking",
        data_size=None,
    ):
        dataset = []
        for data in data_file[uid]['self_data']:
            process_data = {}
            process_data['question'] = data['input']
            process_data['chosen'] = data['output']
            neg_uid = uid
            for i in range(3):
                while neg_uid == uid:
                    neg_uid = random.choice(list(data_file.keys()))
                neg_output = random.choice(data_file[neg_uid]['self_data'])['output']
                process_data['rejected'] = neg_output
                dataset.append(process_data)
            
        dataset = Dataset.from_list(dataset)
        if data_size:    
            dataset = dataset.select(range(data_size))
         
            
        def return_prompt_and_responses(samples) -> Dict[str, str]:
            questions = samples["question"]
            prompts = []
            for question in samples["question"]:
                prompts.append(SYSTEM_PROMPT.format(question))
            return {
                "question": questions,
                "prompt": prompts,
                "chosen": [" " + (s if s is not None else "") for s in samples["matching"]],
                "rejected": [" " + (s if s is not None else "") for s in samples["not_matching"]],
            }

        return dataset

    def get_data_for_caa(
        self,
        data_name="power-seeking",
        split="train",
        data_path="/mnt/20t/msy/shae/data/generation",
    ):
        data_file = os.path.join(
            data_path,
            f"{data_name}.json",
        )
        if os.path.isfile(data_file):
            with open(data_file, 'r') as f:
                data_file = json.load(f)
        else:
            data_file = os.path.join(
                data_path,
                f"{data_name}.pkl",
            )
            with open(data_file, 'r') as f:
                data_file = pickle.load(f)
        return data_file

    def get_data_for_caa_eval(self, data_name, data_path):
        with open(os.path.join(data_path, data_name, 'processed', 'seen_test.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        with open(os.path.join(data_path, data_name, 'processed', 'seen_test_ranked.json'), 'r') as f:
            test_ranked = json.load(f)
            
        return test_data, test_ranked

    