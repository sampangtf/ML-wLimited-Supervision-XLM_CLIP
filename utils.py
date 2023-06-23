import os 
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel, M2M100ForConditionalGeneration, M2M100Tokenizer
import pandas as pd
import numpy as np
import torch

import gc



def sample_k(id_array, exception_index, k=9):
    '''
    Given a list of array which represents indices, randomly sample k subsamples except for exception_index
    :param id_array:
    :param exception_index:
    :param k:
    :return:
    '''
    return np.random.choice(np.delete(id_array, exception_index), k, replace=False) # np.delete might not be the best

def get_captions(pandas_data, image_indices):
    '''
    Given an array of image_index, get the corresponding captions and randomly select one of the captions for each index
    :param pandas_data:
    :param image_indices:
    :param k:
    :return:
    '''

    k = len(image_indices)

    temp_pd_df = pandas_data[pandas_data['image_id'].isin(image_indices)]
    # print("temp_pd_df\n", temp_pd_df)
    return temp_pd_df.reset_index().loc[[i * 5 for i in range(k)] + np.random.choice(5, k)].drop(columns=['index'])

# convert the image_id into a file name
def fix_names(name, fixed_len=12):
    name_str = str(name)

    return '0' * (fixed_len - len(name_str)) + name_str + '.jpg'

# translate text into another langauges
def machine_translate(model, tokenizer, texts: int, src_lang, tgt_lang, device="cpu", verbose=False):  
    tokenizer.src_lang = src_lang
    translated_texts = []
    for text in texts:
        encoded_src = tokenizer(texts, return_tensors="pt").to(device)
        generated_tokens = model.generate(**encoded_src, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
        out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_texts.append(out[0])

        if verbose:
            print("original text: ", text)
            print("translated text: ", out[0])
    return translated_texts
    
def return_topk_rank(model, processor, image, texts, k=5, multiple_images=True, device="cpu"):

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)

    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

    if multiple_images:
        probs = logits_per_image.softmax(dim=0)
        return_text = texts
        topk = probs.reshape(1, -1).topk(k)

    else:
        if len(texts) < k:
            k = len(texts)
        probs = logits_per_image.softmax(dim=1)
        return_text = texts[probs.argmax()]
        topk = probs.topk(k)

    return probs.detach().cpu().numpy().tolist(), return_text, topk



 

def open_images(folder_name, file_name):
    '''
    Returns a list of images, if list contains 1 image, returns just an Image Object, not a list
    :param folder_name:
    :param file_name: a list of file names
    :return:
    '''

    if len(file_name) == 1:
        file_name = fix_names(file_name)
        
        return Image.open(os.path.join(folder_name, file_name))

    else:
        image_list = []

        for file_name_ in file_name:
            # get the name in the right format
            file_name_ = fix_names(file_name_)

            image_list.append(Image.open(os.path.join(folder_name, file_name_)))

        return image_list