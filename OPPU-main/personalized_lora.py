from colorama import Fore, init
import torch
import json

from instruction import build_rag_instruction, get_his, SYS_PROMPT_SINGLE
init(autoreset=True)
task_name_dict = {
    'news_headline': 'LaMP_4',
    'abstract_generation': 'abstract_generation',
    'pwab': 'pwab'
}
   
class PersonalModel:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.task_name = task_name_dict[args.task_name]
        with open(f"../pa_back/data/{self.task_name}/processed/seen_test_ranked.json", 'r') as f:
            self.test_history_list = json.load(f)
     
    def generate(self, problem_instance, k=0):
        p_id = problem_instance['id']
        ranked_his = get_his(self.task_name, str(problem_instance['id']), k, self.test_history_list)
        raw_prompt = problem_instance['input']
        if k > 0:
            if self.args.task_name == 'pwab':
                raw_prompt = build_rag_instruction(self.task_name, 'raw', problem_instance, ranked_his)
            else:
                raw_prompt = build_rag_instruction(self.task_name, 'raw', problem_instance['input'], ranked_his)

        sys_prompt = SYS_PROMPT_SINGLE if self.args.task_name == 'pwab' else ""
        output = self.generate_response_api(raw_prompt, top_k=1, system_message=sys_prompt)

        output_dict = {
            'id': p_id,
            'generation': output,
            'output': problem_instance['output']
        }

        return output_dict

    def build_instruction(self, prompt):
        if self.args.task_name == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.args.task_name == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.args.task_name == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.args.task_name == "news_headline":
            inp = f"Generate a headline for the following article: {prompt}"
            inp += "Please only generate the most suitable one headline, except which no extra text is needed."
        elif self.args.task_name == "LaMP_5":
            inp = f"Generate a title for the following abstract of a paper: {prompt}"
        elif self.args.task_name == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        elif self.args.task_name == "abstract_generation":
            inp = f"Generate an abstract for the title: {prompt}" if not prompt.startswith('Generate') else prompt
        return inp

    def generate_response_api(
        self,
        prompt: str,
        top_k: int,
        max_length: int = 512,
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
        print(Fore.GREEN + full_prompt)

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
        print(Fore.YELLOW + message)

        return message
        
