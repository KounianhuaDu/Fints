from ChatModels import *
from colorama import Fore, init
init(autoreset=True)

from instruction import pretty_product, SYS_PROMPT_SINGLE

class ZeroShot:
    def __init__(self, args):
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
        else:
            raise NotImplementedError
        self.system_message = SYS_PROMPT_SINGLE if self.args.dataset == 'pwab' else ""
        
        
    def generate(self, problem_instance, k=None):
        p_id = problem_instance['id']
        
        if self.args.dataset == 'pwab':
            raw_prompt = self.build_instruction(problem_instance)
        else:
            raw_prompt = self.build_instruction(problem_instance['input'])
        print(Fore.GREEN + raw_prompt)
        output = self.generator.generate_response_api(raw_prompt, top_k=1, system_message=self.system_message)
        print(Fore.YELLOW + output)
        output_dict = {
            'id': p_id,
            'generation': output,
            'output': problem_instance['output']
        }

        return output_dict

    def build_instruction(self, prompt):
        if self.args.dataset == "LaMP_1":
            inp = f"Write an abstract for this title: {prompt}"
        elif self.args.dataset == "LaMP_2":
            inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
        elif self.args.dataset == "LaMP_3":
            inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
        elif self.args.dataset == "LaMP_4":
            inp = f"Generate a headline for the following article: {prompt}"
            inp += "Please only generate the most suitable one headline, except which no extra text is needed."
        elif self.args.dataset == "LaMP_5":
            inp = f"Generate a headline for the following article: {prompt}" if not prompt.startswith('Generate') else prompt
            inp += "Please only generate the most suitable one headline, except which no extra text is needed."
        elif self.args.dataset == "LaMP_6":
            inp = f"Generate a subject for the following email: {prompt}"
        elif self.args.dataset == "abstract_generation":
            inp = f"Generate an abstract for the title {prompt}" if not prompt.startswith('Generate') else prompt
            if self.args.form == 'raw':
                inp += "\nPlease only generate the most suitable one abstract, except which no extra text is needed."
            elif self.args.form == 'python':
                inp += """\nFormat the output in python code like this:
    ```python
    print("Your generated abstract here")
    ```"""
        elif self.args.dataset == "pwab":
            inp = f"USER {prompt['user_id']}: {prompt['input']}"
            if prompt['type'] == 'review':
                product_info = prompt["output"]["product_info"]
                inp += f"\nHere is the product information: {pretty_product(product_info)}"
            # inp += "Please only generate the most suitable one abstract, except which no extra text is needed."
        return inp


        
