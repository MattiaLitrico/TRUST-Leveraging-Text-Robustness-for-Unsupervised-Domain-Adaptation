import os
import torch
import numpy as np
import pdb
import re
import spacy
import yake
from PIL import ImageFilter
import random

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def save_weights(model, e, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    
    torch.save({
            'epochs': e,
            'weights': model.state_dict()}, filename)

def load_checkpoint(weights, cpu=False):
    if not cpu:
        checkpoint = torch.load(weights)
    else:
        checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    
    return checkpoint

def load_weights(model, weights):
    checkpoint = load_checkpoint(weights)
    print("Loading from epoch ", checkpoint['epochs'])
    model.load_state_dict(checkpoint['weights'])

    return model

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r' ',text)

def remove_numbers(text):
    return re.sub(r'[0-9]', ' ', text)

def remove_html(text):
    html=re.compile(r'<,.*?>')
    return html.sub(r' ',text)

def remove_username(text):
    url = re.compile(r'@[A-Za-z0-9_]+')
    return url.sub(r' ',text)

def pre_process_text(text):
    text = remove_URL(text)
    text = remove_numbers(text)
    text = remove_html(text)
    text = remove_username(text)
    text = remove_punctuation(text)
    text = remove_chinese_chars(text)
    return " ".join(text.split())
    
    #words = text.split()
    #words = [word for word in words if len(word) < 20]
    
    return " ".join(words)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]',' ',text)

def remove_chinese_chars(text):
    return re.sub(r'[\u4e00-\u9fff]+',' ',text)

def remove_parenthesis(text):
    return re.sub(r"\((.*?)\)", r"\1", text)

def text_to_inputs(text, tokenizer, MAX_LEN=160):
    text = pre_process_text(text)
    
    inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
    
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    return {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long)
    }

def extract_keywords(text, custom_kw_extractor):
    text = pre_process_text(text)
    #text = remove_punctuation(text)
    #text = remove_chinese_chars(text)
    #text = remove_parenthesis(text)

    text = str(text)
    
    #rake_nltk_var.extract_keywords_from_text(text)
    #keyword_extracted = rake_nltk_var.get_ranked_phrases()
    
    keyword_extracted = custom_kw_extractor.extract_keywords(text)
    keyword_extracted = [x[0] for x in keyword_extracted]
    
    
    return "-".join(keyword_extracted), keyword_extracted