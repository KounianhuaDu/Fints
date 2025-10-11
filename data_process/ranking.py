import torch
from transformers import AutoModel, AutoTokenizer
import json
import tqdm
try:
    from data_utils import batchify
    from data_utils import extract_strings_between_quotes, extract_after_article, extract_after_review, extract_after_paper, add_string_after_title, extract_after_colon, extract_after_abstract, extract_after_description, extract_from_title, pretty_history
except:
    from .data_utils import batchify
    from .data_utils import extract_strings_between_quotes, extract_after_article, extract_after_review, extract_after_paper, add_string_after_title, extract_after_colon, extract_after_abstract, extract_after_description, extract_from_title, pretty_history
import pickle as pkl
import argparse
import os
from copy import deepcopy


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, k, batch_size = 16, return_weight=False, temperature=0.1):
    query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to("cuda:0")
    output_query = contriver(**query_tokens)
    output_query = mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])
    scores = []
    batched_corpus = batchify(corpus, batch_size)
    for batch in batched_corpus:
        tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
        outputs_batch = contriver(**tokens_batch)
        outputs_batch = mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
        temp_scores = output_query.squeeze() @ outputs_batch.T
        scores.extend(temp_scores.tolist())
    topk_values, topk_indices = torch.topk(torch.tensor(scores), k)
    weights = torch.softmax(topk_values / temperature, dim=0)
    if return_weight:
        return [profile[m] for m in topk_indices.tolist()], weights
    return [profile[m] for m in topk_indices.tolist()]

def retrieve_top_k_with_bm25(corpus, profile, query, k):
    tokenized_corpus = [x.split() for x in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    selected_profs = bm25.get_top_n(tokenized_query, profile, n=k)
    return selected_profs

def classification_citation_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["abstract"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    ids = [x['id'] for x in profile]
    extracted = extract_strings_between_quotes(inp)
    query = f'{extracted[1]} {extracted[2]}'
    return corpus, query, ids

def classification_review_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_review(inp)
    return corpus, query, ids

def generation_news_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["text"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_article(inp)
    return corpus, query, ids

def generation_paper_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["abstract"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_colon(inp)
    return corpus, query, ids

def parphrase_tweet_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    query = extract_after_colon(inp)
    ids = [x['id'] for x in profile]
    return corpus, query, ids

def generation_avocado_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_colon(inp)
    return corpus, query, ids

def classification_movies_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["description"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["description"]}' for x in profile]
    query = extract_after_description(inp)
    ids = [x['id'] for x in profile]
    return corpus, query, ids

def abstract_generation_corpus_maker(inp, profile):
    if 'title' in profile[0].keys():
        corpus = [f"{extract_from_title(x['title'])} {x['abstract']}" for x in profile]
    else:
        corpus = [f"{extract_from_title(x['input'])} {x['output']}" for x in profile]
    query = extract_from_title(inp)
    ids = [x['id'] for x in profile]
    return corpus, query, ids

def pwab_corpus_maker(inp, profile):
    corpus = [pretty_history(item, i) for i, item in enumerate(profile)]
     
    return corpus, inp, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default='../../pa_back/data')
    parser.add_argument("--task", required = True)
    parser.add_argument("--ranker", default='contriever')
    parser.add_argument("--batch_size", type = int, default=16)
    parser.add_argument("--use_date", action='store_true')
    parser.add_argument("--contriever_checkpoint", default="/inspire/hdd/global_user/zhangweinan-24046/contriever")

    args = parser.parse_args()
    task = args.task
    ranker = args.ranker
    
    #Preload the model to avoid repeated loading.
    if ranker == "contriever":
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever", cache_dir=args.contriever_checkpoint)
        contriver = AutoModel.from_pretrained("facebook/contriever", cache_dir=args.contriever_checkpoint).to("cuda:0")
        contriver.eval()
        print('Contriever loaded.')
    
    to_be_ranked = ['train.pkl']
    # to_be_ranked = ["remain_train.pkl"]
    
    for file in to_be_ranked:
        with open(os.path.join(args.data, args.task, 'processed', file), 'rb') as f:
            target_data = pkl.load(f)
        rank_dict = dict()
        for user_id, samples in target_data.items():
            for data in tqdm.tqdm(samples):
                inp = data['input']
                if args.task == 'LaMP_4':
                    profile = data['profile']
                elif args.task == 'pwab':
                    profile = data['history']
                    profile = [item for item in profile if item['review']['timestamp'] < data['timestamp']]
                else:
                    samples_cp = deepcopy(samples)
                    samples_cp.remove(data)
                    profile = samples_cp
                if task == "LaMP_1":
                    corpus, query, ids = classification_citation_query_corpus_maker(inp, profile, args.use_date)
                elif task == "LaMP_3":
                    corpus, query, ids = classification_review_query_corpus_maker(inp, profile, args.use_date)
                elif task == "LaMP_2":
                    corpus, query = classification_movies_query_corpus_maker(inp, profile, args.use_date)
                elif task == "LaMP_4":
                    corpus, query, ids = generation_news_query_corpus_maker(inp, profile, args.use_date)
                elif task == "LaMP_5":
                    corpus, query, ids = generation_paper_query_corpus_maker(inp, profile, args.use_date)
                elif task == "LaMP_7":
                    corpus, query, ids = parphrase_tweet_query_corpus_maker(inp, profile, args.use_date)
                elif task == "LaMP_6":
                    corpus, query, ids = generation_avocado_query_corpus_maker(inp, profile, args.use_date)
                elif task == "abstract_generation":
                    corpus, query, ids = abstract_generation_corpus_maker(inp, profile)
                elif task == "pwab":
                    corpus, query, ids = pwab_corpus_maker(inp, profile)
                
                if ranker == "contriever":
                    randked_profile = retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, len(profile), args.batch_size)
                elif ranker == "bm25":
                    randked_profile = retrieve_top_k_with_bm25(corpus, profile, query, len(profile))
                elif ranker == "recency":
                    profile = sorted(profile, key=lambda x: tuple(map(int, str(x['date']).split("-"))))
                    randked_profile = profile[::-1]

                rank_dict[data['id']] = randked_profile

        out_name = file.split('.')[0] + '_ranked.json'
        with open(os.path.join(args.data, args.task, 'processed', out_name), 'w') as f:
            json.dump(rank_dict, f, indent=4)
        
        print(f"{out_name} generated.")