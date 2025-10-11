import json
import zipfile
import glob
import os
import shutil
import evaluate
from rouge import Rouge
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

def postprocess_text_classification(preds, labels):
    preds = [str(pred).strip() for pred in preds]
    labels = [str(label).strip() for label in labels]
    return preds, labels

def postprocess_text_generation(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def create_metric_f1_accuracy(all_labels):
    f1_metric = evaluate.load("f1", cache_dir="./evaluate_metrics/f1")
    accuracy_metric = evaluate.load("accuracy", cache_dir="./evaluate_metrics/accuracy")
    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            return -1
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        result_acc = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels, labels=list(range(len(all_labels))), average = "macro")
        result = {"accuracy" : result_acc["accuracy"], "f1" : result_f1["f1"]}
        return result
    return compute_metrics

def create_metric_mae_rmse():
    mse_metric = evaluate.load("mse", cache_dir="./evaluate_metrics/mse")
    mae_metric = evaluate.load("mae", cache_dir="./evaluate_metrics/mae")
    def create_mapping(x, y):
        try:
            return float(x)
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x,y) for x,y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x,x) for x in decoded_labels]
        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared = False)
        result = {"MAE" : result_mae["mae"], "RMSE" : result_rmse["mse"]}
        return result
    return compute_metrics

def create_metric_rouge():
    rouge_metric = evaluate.load("rouge", cache_dir='./evaluate_metrics/rouge')
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"rouge-1" : result_rouge["rouge1"], "rouge-L" : result_rouge["rougeL"]}
        return result
    return compute_metrics

def evaluate_task(predictions_dir):
    if predictions_dir.endswith('.json'):
        with open(predictions_dir, 'rb') as f:
            data = json.load(f)
        
        task_name = data['task']
        lines = data['golds']
        
        preds = []
        golds = []
        for line in lines:
            preds.append(regularize(line['generation']))
            golds.append(line['output'])
    else:
        data_files = os.listdir(predictions_dir)
        preds = []
        golds = []
        for data_file in data_files:
            with open(os.path.join(predictions_dir, data_file), 'r') as f:
                data = json.load(f)
                
            task_name = data['task']
            lines = data['golds']
            for line in lines:
                line['generation'].replace("<|begin_of_text|>", "")
                preds.append(regularize(line['generation']))
                golds.append(line['output'])
                
    if task_name in ["LaMP_1", "LaMP_2"]:
        metric = create_metric_f1_accuracy(self._get_labels(task_name))
    elif task_name == "LaMP_3":
        metric = create_metric_mae_rmse()
    else:
        metric = create_metric_rouge()
    
    return metric(preds, golds)
    
class LaMPEvaluation(object):
    
    def __init__(self, all_golds_zip_file_addr = None, single_gold_json_file_addr = None, extract_addr = "./tmp") -> None:
        assert all_golds_zip_file_addr or single_gold_json_file_addr, "The golds should be provided for all datasets or at least one."
        assert not (all_golds_zip_file_addr and single_gold_json_file_addr), "The golds should be provided using zip file or json file not both."
        self.tasks_golds = dict()
        self.extract_addr = extract_addr
        self.evaluate_all_is_possible = False
        if all_golds_zip_file_addr:
            os.makedirs(self.extract_addr, exist_ok=True)
            with zipfile.ZipFile(all_golds_zip_file_addr, 'r') as zobj:
                zobj.extractall(path = extract_addr)
            for file_addr in glob.glob(os.path.join(self.extract_addr, "**/*.json"), recursive=True):
                with open(file_addr) as file:
                    task = json.load(file)
                    self.tasks_golds[task['task']] = task['golds']
            self._empty_dir(self.extract_addr)
            self.evaluate_all_is_possible = True
        '''if single_gold_json_file_addr:
            with open(single_gold_json_file_addr) as file:
                    task = json.load(file)
                    self.tasks_golds[task['task']] = task['golds']''' 
        
        if single_gold_json_file_addr:
            with open(single_gold_json_file_addr) as file:
                task = json.load(file)
                self.tasks_golds['LaMP_4'] = [{'id': sample['id'], 'output': sample['output']} for sample in task]
    
    def _empty_dir(self, directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def _get_all_gold_ids(self, task_name):
        return set([sample['id'] for sample in self.tasks_golds[task_name]])
    
    def _get_all_ids(self, input):
        return set([sample['id'] for sample in input])
    
    def evaluate_all(self, predicts_zipfile_addr):
        assert self.evaluate_all_is_possible, "You did not provide golds for all tasks."
        with zipfile.ZipFile(predicts_zipfile_addr, 'r') as zobj:
            zobj.extractall(path = self.extract_addr)
        results_raw = dict()
        all_task_names = set()
        for file_addr in glob.glob(os.path.join(self.extract_addr, "**/*.json"), recursive=True):
            with open(file_addr) as file:
                preds = json.load(file)
            all_task_names.add(preds['task'])
            results_raw[preds['task']] = self._evaluate_task(preds['golds'], preds['task'])
        self._empty_dir(self.extract_addr)
        assert len(all_task_names) == 7, "The provided results do not cover all the tasks in the benchmark."
        return results_raw

    def evaluate_task(self, predicts_json_addr, task_name):
        with open(predicts_json_addr) as file:
            preds = json.load(file)
        assert preds['task'] == task_name, "The provided task_name and the results do not match."
        assert preds['task'] in self.tasks_golds.keys(), "The provided golds cannot be used to evaluate this task."
        return self._evaluate_task(preds['golds'], task_name)

    def _evaluate_task(self, predictions, task_name):
        golds_dict = {y['id']:y['output'] for y in self.tasks_golds[task_name]}
        preds_dict = {x['id']: self.regularize(x['output']) for x in predictions}
        
        gold_ids = self._get_all_gold_ids(task_name)
        pred_ids = self._get_all_ids(predictions)

        assert gold_ids == pred_ids, "Predictions ids and gold ids do not match."

        if task_name in ["LaMP_1", "LaMP_2"]:
            metric = create_metric_f1_accuracy(self._get_labels(task_name))
        elif task_name == "LaMP_3":
            metric = create_metric_mae_rmse()
        else:
            metric = create_metric_rouge()
        
        gold_ids = list(gold_ids)
        golds = [golds_dict[id] for id in gold_ids]
        preds = [preds_dict[id] for id in gold_ids]
        print(preds)
        return metric(preds, golds)
    
    def _get_labels(self, task_name):
        if task_name == "LaMP_1":
            return ["[1]", "[2]"]
        elif task_name == "LaMP_2":
            return ['sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 'dark comedy', 'classic', 'psychology', 'fantasy', 'romance', 'thought-provoking', 'social commentary', 'violence', 'true story']
        elif task_name == "LaMP_3":
            return ["1", "2", "3", "4", "5"]
        else:
            raise ValueError("Invalid task_name")
        
def regularize(text):
    lines = text.split('\n')
    res = ""
    for line in lines:
        if len(line) > 1 and 'headline' not in line and 'title' not in line and 'abstract' not in line:
            return line  
    return res
    
def calculate_reward(task, action, observation):
    res = [0, 0.0]
    if task['type'] == 'search':
        if action == 'search_product_by_query':
            res[0] = 1
        else :
            return res
        target_asin = task['output']['product_info']['parent_asin']
        if isinstance(observation, list):
            for i in range(len(observation)):
                if target_asin in observation[i]:
                    res[1] = 1 - i/len(observation)
                    break

    elif task['type'] == 'recommend':
        if action == 'get_recommendations_by_history':
            res[0] = 1
        else :
            return res
        target_asin = task['output']['product_info']['parent_asin']
        if isinstance(observation, list):
            for i in range(len(observation)):
                if target_asin in observation[i]:
                    res[1] = 1 - i/len(observation)
                    break

    elif task['type'] == 'review':
        if action == 'add_product_review':
            res[0] = 1
        else :
            return res
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