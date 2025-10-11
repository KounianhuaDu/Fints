from collections import Counter
import pickle as pkl
import json
import os
import random
from colorama import Fore, init
from collections import defaultdict
import pandas as pd
import glob
init(autoreset=True)

def process_data(root_dir, dataset, threshold):
    with open(os.path.join(root_dir, dataset, 'train/train_questions.json'), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(root_dir, dataset, 'train/train_outputs.json'), 'r') as f:
        train_output = json.load(f)
        
    with open(os.path.join(root_dir, dataset, 'valid/dev_questions.json'), 'r') as f:
        test_data = json.load(f)  
    with open(os.path.join(root_dir, dataset, 'valid/dev_outputs.json'), 'r') as f:
        test_output = json.load(f)

    # key: user id; value: total history length.
    train_uid_set = defaultdict(int)
    test_uid_set = defaultdict(int)
    
    # samples indexed by user id.
    train_samples = defaultdict(list)
    test_samples = defaultdict(list)
    
    for sample, out in zip(train_data, train_output['golds']):
        assert sample['id'] == out['id'] 
        line = {}
        line['id'] = sample['id']
        line['input'] = sample['input']
        line['profile'] = sample['profile']
        line['user_id'] = sample['user_id']
        line['output'] = out['output']
        train_uid_set[sample['user_id']] += len(sample['profile'])
        train_samples[sample['user_id']].append(line)
    
    for sample, out in zip(test_data, test_output['golds']):
        assert sample['id'] == out['id'] 
        line = {}
        line['id'] = sample['id']
        line['input'] = sample['input']
        line['profile'] = sample['profile']
        line['user_id'] = sample['user_id']
        line['output'] = out['output']
        test_uid_set[sample['user_id']] += len(sample['profile'])
        test_samples[sample['user_id']].append(line)
        
    interaction_users_set = set(train_uid_set.keys()) & set(test_uid_set.keys())
    filtered_interaction_users_set = random.sample(interaction_users_set, threshold)
    
    unseen_users_test = set(test_uid_set.keys()).difference(filtered_interaction_users_set)
    filtered_unseen_users_test = random.sample(unseen_users_test, 50)
    
    remaining_train = defaultdict(list)
    train = defaultdict(list)
    seen_test = defaultdict(list)
    unseen_test = defaultdict(list)
    
    for uid, samples in train_samples.items():
        if uid in filtered_interaction_users_set:
            train[uid] = samples
        else:
            remaining_train[uid] = samples
    
    for uid, samples in test_samples.items():
        if uid in filtered_interaction_users_set:
            seen_test[uid] = samples
        elif uid in filtered_unseen_users_test:
            unseen_test[uid] = samples
    
    os.makedirs(os.path.join(root_dir, dataset, 'processed'), exist_ok=True)
    
    with open(os.path.join(root_dir, dataset, 'processed', 'train.pkl'), 'wb') as f:
        pkl.dump(train, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'remain_train.pkl'), 'wb') as f:
        pkl.dump(remaining_train, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'seen_test.pkl'), 'wb') as f:
        pkl.dump(seen_test, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'unseen_test.pkl'), 'wb') as f:
        pkl.dump(unseen_test, f)
        
    train_sample_num = [train_uid_set[uid] for uid in list(train.keys())]
    remain_train_sample_num = [train_uid_set[uid] for uid in list(remaining_train.keys())]
    seen_test_sample_num = [test_uid_set[uid] for uid in list(seen_test.keys())]
    unseen_test_sample_num = [test_uid_set[uid] for uid in list(unseen_test.keys())]
    
    qualified_train = [int(num>200) for num in train_sample_num]
    qualified_train_remain = [int(num>200) for num in remain_train_sample_num]
    qualified_seen_test = [int(num>200) for num in seen_test_sample_num]
    qualified_unseen_test = [int(num>200) for num in unseen_test_sample_num]
    
    print(Fore.GREEN + f"# User of Train: {len(list(train.keys()))}, Qualified: {sum(qualified_train)}")
    print(Fore.GREEN + f"# User of Remain Train: {len(list(remaining_train.keys()))}, Qualified: {sum(qualified_train_remain)}")
    print(Fore.GREEN + f"# User of Seen Test: {len(list(seen_test.keys()))}, Qualified: {sum(qualified_seen_test)}")
    print(Fore.GREEN + f"# User of Unseen Test: {len(list(unseen_test.keys()))}, Qualified: {sum(qualified_unseen_test)}")
    
    print(Fore.GREEN + f"# Historical Samples of Train: {sum(train_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Remain Train: {sum(remain_train_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Seen Test: {sum(seen_test_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Unseen Test: {sum(unseen_test_sample_num)}")
    
def process_data_parquet(root_dir, dataset, threshold):
    all_files = [f for f in os.listdir(os.path.join(root_dir, dataset)) if 'train' in f]
    train_data = []
    for file in all_files:
        df = pd.read_parquet(os.path.join(root_dir, dataset, file))
        train_data.extend(df.to_dict(orient='records'))
    
    all_files = [f for f in os.listdir(os.path.join(root_dir, dataset)) if 'test' in f]
    test_data = []
    for file in all_files:
        df = pd.read_parquet(os.path.join(root_dir, dataset, file))
        test_data.extend(df.to_dict(orient='records'))
        
    # key: user id; value: total history length.
    train_uid_set = defaultdict(int)
    test_uid_set = defaultdict(int)
    
    # samples indexed by user id.
    train_samples = defaultdict(list)
    test_samples = defaultdict(list)
    
    pid = 0
    for sample in train_data:
        line = {}
        line['input'] = sample['input']
        line['user_id'] = sample['name']
        line['output'] = sample['output']
        line['id'] = pid
        pid += 1
        train_uid_set[sample['name']] += len(sample['profile'])
        train_samples[sample['name']].append(line)
        for s in sample['profile']:
            line = {}
            line['input'] = s['title']
            line['user_id'] = sample['name']
            line['output'] = s['abstract']
            line['id'] = s['id']
            train_samples[sample['name']].append(line)
    
    for sample in test_data:
        line = {}
        line['input'] = sample['input']
        line['profile'] = sample['profile']
        line['user_id'] = sample['name']
        line['output'] = sample['output']
        line['id'] = pid
        pid += 1
        test_uid_set[sample['name']] += len(sample['profile'])
        test_samples[sample['name']].append(line)
        
    interaction_users_set = set(train_uid_set.keys()) & set(test_uid_set.keys())
    filtered_interaction_users_set = random.sample(interaction_users_set, threshold)
    
    unseen_users_test = set(test_uid_set.keys()).difference(filtered_interaction_users_set)
    filtered_unseen_users_test = random.sample(unseen_users_test, 50)
    
    remaining_train = defaultdict(list)
    train = defaultdict(list)
    seen_test = defaultdict(list)
    unseen_test = defaultdict(list)
    
    for uid, samples in train_samples.items():
        if uid in filtered_interaction_users_set:
            print(Fore.YELLOW + f"User {uid} has {train_uid_set[uid] + 1} samples.")
            train[uid] = samples
        else:
            remaining_train[uid] = samples
    for uid, samples in test_samples.items():
        if uid in filtered_interaction_users_set:
            seen_test[uid] = samples
        elif uid in filtered_unseen_users_test:
            unseen_test[uid] = samples
            
    os.makedirs(os.path.join(root_dir, dataset, 'processed'), exist_ok=True)
    
    with open(os.path.join(root_dir, dataset, 'processed', 'train.pkl'), 'wb') as f:
        pkl.dump(train, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'remain_train.pkl'), 'wb') as f:
        pkl.dump(remaining_train, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'seen_test.pkl'), 'wb') as f:
        pkl.dump(seen_test, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'unseen_test.pkl'), 'wb') as f:
        pkl.dump(unseen_test, f)
            
    train_sample_num = [train_uid_set[uid] for uid in list(train.keys())]
    remain_train_sample_num = [train_uid_set[uid] for uid in list(remaining_train.keys())]
    seen_test_sample_num = [test_uid_set[uid] for uid in list(seen_test.keys())]
    unseen_test_sample_num = [test_uid_set[uid] for uid in list(unseen_test.keys())]
    
    qualified_train = [int(num>200) for num in train_sample_num]
    qualified_train_remain = [int(num>200) for num in remain_train_sample_num]
    qualified_seen_test = [int(num>200) for num in seen_test_sample_num]
    qualified_unseen_test = [int(num>200) for num in unseen_test_sample_num]
            
    print(Fore.GREEN + f"# User of Train: {len(list(train.keys()))}, Qualified: {sum(qualified_train)}")
    print(Fore.GREEN + f"# User of Remain Train: {len(list(remaining_train.keys()))}, Qualified: {sum(qualified_train_remain)}")
    print(Fore.GREEN + f"# User of Seen Test: {len(list(seen_test.keys()))}, Qualified: {sum(qualified_seen_test)}")
    print(Fore.GREEN + f"# User of Unseen Test: {len(list(unseen_test.keys()))}, Qualified: {sum(qualified_unseen_test)}")
    
    print(Fore.GREEN + f"# Historical Samples of Train: {sum(train_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Remain Train: {sum(remain_train_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Seen Test: {sum(seen_test_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Unseen Test: {sum(unseen_test_sample_num)}")
    
def process_data_pwab(root_dir, dataset, threshold):
    FOLDER_PATH = os.path.join(root_dir, dataset, 'data')
    def merge_json_files(file_pattern):
        merged_data = {}

        for json_file in glob.glob(file_pattern):
            with open(json_file, 'r') as f:
                data = json.load(f)
                merged_data.update(data)  
            #print(f"Loaded {json_file}")
        
        return merged_data

    all_products = merge_json_files(os.path.join(FOLDER_PATH, "all_products_part_*.json"))
    user_history = merge_json_files(os.path.join(FOLDER_PATH, "user_history_part_*.json"))
    all_data = json.load(open(os.path.join(FOLDER_PATH, "user_instructions.json")))

    train_data = all_data['train']
    test_data = all_data['test']
    # key: user id; value: total history length.
    train_uid_set = defaultdict(int)
    test_uid_set = defaultdict(int)
    
    # samples indexed by user id.
    train_samples = defaultdict(list)
    test_samples = defaultdict(list)
    
    pid = 0
    for sample in train_data:
        line = {}
        line['input'] = sample['task']
        line['user_id'] = sample['user_id']
        line['output'] = sample['target']
        line['id'] = pid
        pid += 1
        line['type'] = sample['type']
        line['history'] = user_history[line['user_id']]
        line['timestamp'] = sample['timestamp']
        train_uid_set[sample['user_id']] += len(user_history[line['user_id']])
        train_samples[sample['user_id']].append(line)
    
    for sample in test_data:
        line = {}
        line['input'] = sample['task']
        line['user_id'] = sample['user_id']
        line['output'] = sample['target']
        line['id'] = pid
        pid += 1
        line['type'] = sample['type']
        line['history'] = user_history[line['user_id']]
        line['timestamp'] = sample['timestamp']
        test_uid_set[sample['user_id']] += len(user_history[line['user_id']])
        test_samples[sample['user_id']].append(line)
        
    interaction_users_set = set(train_uid_set.keys()) & set(test_uid_set.keys())
    qualified_users_set = [uid for uid in train_uid_set.keys() if len(train_samples[uid]) >= 5]
    print(len(qualified_users_set))
    filtered_interaction_users_set = random.sample(qualified_users_set, threshold)
    
    unseen_users_test = set(test_uid_set.keys()).difference(filtered_interaction_users_set)
    filtered_unseen_users_test = random.sample(unseen_users_test, 50)
    
    remaining_train = defaultdict(list)
    train = defaultdict(list)
    seen_test = defaultdict(list)
    unseen_test = defaultdict(list)
    
    for uid, samples in train_samples.items():
        if uid in filtered_interaction_users_set:
            print(Fore.YELLOW + f"User {uid} has {len(samples)} samples.")
            train[uid] = samples
        else:
            remaining_train[uid] = samples
    for uid, samples in test_samples.items():
        if uid in filtered_interaction_users_set:
            seen_test[uid] = samples
        elif uid in filtered_unseen_users_test:
            unseen_test[uid] = samples
            
    os.makedirs(os.path.join(root_dir, dataset, 'processed'), exist_ok=True)
    
    with open(os.path.join(root_dir, dataset, 'processed', 'train.pkl'), 'wb') as f:
        pkl.dump(train, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'remain_train.pkl'), 'wb') as f:
        pkl.dump(remaining_train, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'seen_test.pkl'), 'wb') as f:
        pkl.dump(seen_test, f)
    with open(os.path.join(root_dir, dataset, 'processed', 'unseen_test.pkl'), 'wb') as f:
        pkl.dump(unseen_test, f)
            
    train_sample_num = [train_uid_set[uid] for uid in list(train.keys())]
    remain_train_sample_num = [train_uid_set[uid] for uid in list(remaining_train.keys())]
    seen_test_sample_num = [test_uid_set[uid] for uid in list(seen_test.keys())]
    unseen_test_sample_num = [test_uid_set[uid] for uid in list(unseen_test.keys())]
    
    qualified_train = [int(num>200) for num in train_sample_num]
    qualified_train_remain = [int(num>200) for num in remain_train_sample_num]
    qualified_seen_test = [int(num>200) for num in seen_test_sample_num]
    qualified_unseen_test = [int(num>200) for num in unseen_test_sample_num]
            
    print(Fore.GREEN + f"# User of Train: {len(list(train.keys()))}, Qualified: {sum(qualified_train)}")
    print(Fore.GREEN + f"# User of Remain Train: {len(list(remaining_train.keys()))}, Qualified: {sum(qualified_train_remain)}")
    print(Fore.GREEN + f"# User of Seen Test: {len(list(seen_test.keys()))}, Qualified: {sum(qualified_seen_test)}")
    print(Fore.GREEN + f"# User of Unseen Test: {len(list(unseen_test.keys()))}, Qualified: {sum(qualified_unseen_test)}")
    
    print(Fore.GREEN + f"# Historical Samples of Train: {sum(train_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Remain Train: {sum(remain_train_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Seen Test: {sum(seen_test_sample_num)}")
    print(Fore.GREEN + f"# Historical Samples of Unseen Test: {sum(unseen_test_sample_num)}")
    
process_data_parquet('../../pa_back/data', 'abstract_generation', 1000)

