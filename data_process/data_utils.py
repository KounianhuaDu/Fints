import json

HISTORY_PROMPT = '''
MEMORY <NUM>:

Product:
- Title: <TITLE>
- Parent Asin: <PARENT_ASIN>
- Main Category: <MAIN_CATEGORY>
- Average Rating: <AVERAGE_RATING>
- Rating Number: <RATING_NUMBER>
- Price: <PRICE>
- Store: <STORE>
- Details: <DETAILS>
- Description: <DESCRIPTION>
- Features: <FEATURES>

Review:
- Rating: <RATING>
- Text: <TEXT>
- Timestamp: <TIMESTAMP>

'''

MINI_HISTORY_PROMPT = '''
MEMORY <NUM>:
Product:
- Title: <TITLE>
- Parent Asin: <PARENT_ASIN>
- Main Category: <MAIN_CATEGORY>
Review:
- Rating: <RATING>
- Text: <TEXT>
- Timestamp: <TIMESTAMP>
'''

def extract_strings_between_quotes(input_string):
    output_list = []
    inside_quotes = False
    current_string = ''
    
    for char in input_string:
        if char == '"' and not inside_quotes:
            inside_quotes = True
        elif char == '"' and inside_quotes:
            inside_quotes = False
            output_list.append(current_string)
            current_string = ''
        elif inside_quotes:
            current_string += char
    
    return output_list

def extract_after_article(input_string):
    article_index = input_string.find('article:')
    if article_index == -1:
        return None
    return input_string[article_index + len('article:'):].strip()

def extract_after_description(input_string):
    article_index = input_string.find('description:')
    if article_index == -1:
        return None
    return input_string[article_index + len('description:'):].strip()


def extract_after_review(input_string):
    article_index = input_string.find('review:')
    if article_index == -1:
        return None
    return input_string[article_index + len('review:'):].strip()

def extract_after_paper(input_string):
    article_index = input_string.find('paper:')
    if article_index == -1:
        return None
    return input_string[article_index + len('paper:'):].strip()

def extract_after_abstract(input_string):
    article_index = input_string.find('abstract:')
    if article_index == -1:
        return None
    return input_string[article_index + len('abstract:'):].strip()

def extract_after_colon(input_string):
    article_index = input_string.find(':')
    if article_index == -1:
        return None
    return input_string[article_index + len(':'):].strip()

def extract_from_title(input_string):
    if '\"' in input_string:
        input_string = input_string.split('\"')[1]
    return input_string


def add_string_after_title(original_string, string_to_add):
    title_index = original_string.find("title")
    
    if title_index == -1:
        return original_string
    
    return original_string[:title_index+5] + ", and " + string_to_add + original_string[title_index+5:]

def batchify(lst, batch_size):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

def pretty_history(item, num):
    res = HISTORY_PROMPT.replace("<TITLE>", item['product_info']['title'])
    res = res.replace("<PARENT_ASIN>", item['product_info']['parent_asin'])
    res = res.replace("<AVERAGE_RATING>", str(item['product_info']['average_rating']))
    res = res.replace("<RATING_NUMBER>", str(item['product_info']['rating_number']))
    res = res.replace("<PRICE>", str(item['product_info']['price']))
    res = res.replace("<STORE>", str(item['product_info']['store']))
    res = res.replace("<DETAILS>", json.dumps(item['product_info']['details']))
    res = res.replace("<DESCRIPTION>", str(item['product_info']['description']))
    res = res.replace("<FEATURES>", str(item['product_info']['features']))
    res = res.replace("<MAIN_CATEGORY>", str(item['product_info']['main_category']))
    
    res = res.replace("<RATING>", str(item['review']['rating']))
    res = res.replace("<TEXT>", item['review']['text'])
    res = res.replace("<TIMESTAMP>", str(item['review']['timestamp']))
    res = res.replace("<NUM>", str(num))
    return res

def mini_pretty_history(item, num):
    res = MINI_HISTORY_PROMPT.replace("<TITLE>", item['product_info']['title'])
    res = res.replace("<PARENT_ASIN>", item['product_info']['parent_asin'])
    res = res.replace("<MAIN_CATEGORY>", str(item['product_info']['main_category']))
    res = res.replace("<RATING>", str(item['review']['rating']))
    res = res.replace("<TEXT>", item['review']['text'])
    res = res.replace("<TIMESTAMP>", str(item['review']['timestamp']))
    res = res.replace("<NUM>", str(num))
    return res