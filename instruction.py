import random
import json

try:
    from data_process import pretty_history, mini_pretty_history
    from pwab import functions, data

    functions_info = {tool.__name__: tool.__info__ for tool in functions}
except:
    pass

PRODUCT_PROMPT = '''
- Title: <TITLE>
- Parent Asin: <PARENT_ASIN>
- Average Rating: <AVERAGE_RATING>
- Rating Number: <RATING_NUMBER>
- Price: <PRICE>
- Store: <STORE>
- Features: <FEATURES>
- Description: <DESCRIPTION>
- Details: <DETAILS>
- Category: <CATEGORY>
'''

SYS_PROMPT_SINGLE = '''As a personalized shopping agent, you can help users search for products, recommend products or complete their reviews.

Rules:
- The user will provide user_id and a request.
- You need to use the most appropriate tool to find the product or fill the review that matches the user's request.
- For different requests, you may need to use different tools. Correct tool selection and better tool input will help you get better results.
- You are not allowed to interact with the user. Make the best tool call based on the user's request.
- You are only allowed to make one try. Once you make a tool call to search_product_by_query, get_recommendations_by_history or add_product_review, task will be over.
- The evaluation will be based on the tool selection and ranking of the target product in search and recommendation tasks, and the similarity of the review in the review task.
- If memory about the user is provided, you should use it to help you formulate better tool calls.

You have access to the following tools:

1. add_product_review(review: string)
   - Description: Add a full text review. The review should be a string. Do not include extra parameters.

2. get_recommendations_by_history(product_sequence: list of strings)
   - Description: Get recommended products based on an input sequence of product asins. 
   - Purpose: Generate product recommendations from the user's product history.
   - Input must be an ordered list of product ASINs from your memory, e.g. ["B07S1D3YTW", "B074V8R6NL", "B07PPPT1NG"].
   - The sequence should follow the time order, with the most recent product asin at the end. 
   - Maximum length is 30.

3. search_product_by_query(query: string)
   - Description: Search for products by a query string. The information of the top 10 products will be returned.

--------------------
### Rules for output:
- You MUST always call exactly one tool from the list above. 
- The output MUST be valid JSON with the following schema:

```json
{
  "tool_call": {
    "name": "<tool name>",
    "arguments": {
      ... parameters here ...
    }
  }
}
```

- Do not include any text outside of the JSON.
- Do not invent new tools or parameters.
- Choose the most relevant tool based on the user's request.

### Examples:
Your Memory of This User:
MEMORY 1: Product: 
- Parent Asin: B001
MEMORY 2: Product: 
- Parent Asin: B002
MEMORY 3: Product: 
- Parent Asin: B003
User: Please recommend something new.
Assistant:
{
  "tool_call": {
    "name": "get_recommendations_by_history",
    "arguments": {
      "product_sequence": ["B001","B002","B003"]
    }
  }
}
'''

SYS_PROMPT_POS = '''As a personalized shopping agent, you can help users search for products, recommend products or complete their reviews.

Rules:
- The user will provide user_id and a request.
- You need to use the most appropriate tool to find the product or fill the review that matches the user's request.
- For different requests, you may need to use different tools. Correct tool selection and better tool input will help you get better results.
- You are not allowed to interact with the user. Make the best tool call based on the user's request.
- You are only allowed to make one try. Once you make a tool call to search_product_by_query, get_recommendations_by_history or add_product_review, task will be over.
- The evaluation will be based on the tool selection and ranking of the target product in search and recommendation tasks, and the similarity of the review in the review task.
- If memory about the user is provided, you should use it to help you formulate better tool calls.

--------------------
### Rules for output:
- You MUST always call exactly the tool given by USER. 
- The output MUST be valid JSON with the following schema:

```json
{
  "tool_call": {
    "name": "<tool name>",
    "arguments": {
      ... parameters here ...
    }
  }
}
```

- Do not include any text outside of the JSON.
- Do not invent new tools or parameters.
'''

def get_his(dataset, p_id, k, calibration_ranked, random_sample=False):
    if random_sample:
        ranked_profiles = random.sample(calibration_ranked[p_id], k)
    else:
        ranked_profiles = calibration_ranked[p_id][:k]
    
    q_a_history = []
    for idx, sample in enumerate(ranked_profiles):
        if dataset in ['pwab', 'pwab_pos']:
            line = pretty_history(sample, idx)
        elif dataset == "LaMP_4":
            line = f"Historical sample {idx}:\n Q: {sample['text']}. \n A: {sample['title']}."
        elif dataset == "LaMP_5":
            line = f"Historical sample {idx}:\n Q: {sample['abstract']}. \n A: {sample['title']}."
        elif dataset == "abstract_generation":
            line = f"Historical sample {idx}:\n Q: {sample['title']}. \n A: {sample['abstract']}."
        else:
            raise NotImplementedError
        q_a_history.append(line)
    q_a_history = '\n'.join(q_a_history)
    return q_a_history
    
def build_rag_instruction(dataset, form, prompt, his):
    if dataset == "LaMP_1":
        inp = f"Write an abstract for this title: {prompt}"
    elif dataset == "LaMP_2":
        inp = f"Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {prompt}"
    elif dataset == "LaMP_3":
        inp = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {prompt}"
    elif dataset == "LaMP_4":  
        inp = f"Generate a headline for the following article: {prompt}" if not prompt.startswith('Generate') else prompt
        if his:
            inp += f"\nFor your reference, here are the user's past QA pairs:\n {his}\n"
        if form == 'raw':
            inp += "Please only generate the most suitable one headline, except which no extra text is needed."
        elif form == 'json':
            inp += """\nFormat the output in json format like this:  
```json
{"headline": Your generated headline here}  
```  """
        elif form == 'python':
            inp += """\nFormat the output in python code like this:
```python
print("Your generated headline here")
```"""
    elif dataset == "LaMP_5":
        inp = f"Generate a title for the following abstract of a paper: {prompt}"
        if his:
            inp += f"For your reference, here are the user's past QA pairs:\n {his}\n"
        if form == 'raw':
            inp += "Please only generate the most suitable one abstract, except which no extra text is needed."
        elif form == 'json':
            inp += """\nFormat the output in json format like this:  
```json
{"headline": Your generated title here}  
```  """
        elif form == 'python':
            inp += """\nFormat the output in python code like this:
```python
print("Your generated title here")
```"""
    elif dataset == "LaMP_6":
        inp = f"Generate a subject for the following email: {prompt}"
    elif dataset == "abstract_generation":
        inp = f"Generate an abstract for the title: {prompt}" if not prompt.startswith('Generate') else prompt
        inp += f"\nFor your reference, here are the user's past QA pairs:\n {his}\n"
        if form == 'raw':
            inp += "\nPlease only generate the most suitable one abstract, except which no extra text is needed."
        elif form == 'python':
            inp += """\nFormat the output in python code like this:
```python
print("Your generated abstract here")
```"""
    elif dataset in ['pwab', 'pwab_pos']:
        # inp = SYS_PROMPT_SINGLE
        inp = f"Your Memory of This User:\n\n{his}"
        inp += f"\nUSER {prompt['user_id']}: {prompt['input']}"
        if prompt['type'] == 'review':
            product_info = prompt["output"]["product_info"]
            inp += f"\nHere is the product information: {pretty_product(product_info)}"          
        inp += f"\nUSER {prompt['user_id']}: {prompt['input']}"
            
    return inp

def pretty_product(product: dict) -> str:
    # print(product)
    res = PRODUCT_PROMPT.replace('<CATEGORY>', str(product['main_category']))
    res = res.replace('<TITLE>', product['title'])
    res = res.replace('<AVERAGE_RATING>', str(product['average_rating']))
    res = res.replace('<RATING_NUMBER>', str(product['rating_number']))
    features = str(product['features'])

    res = res.replace('<FEATURES>', features)
    description = str(product['description'])
    res = res.replace('<DESCRIPTION>', description)
    res = res.replace('<PRICE>', str(product['price']))
    res = res.replace('<STORE>', str(product['store']))
    res = res.replace('<DETAILS>', json.dumps(product['details']))
    res = res.replace('<PARENT_ASIN>', product['parent_asin'])
    return res