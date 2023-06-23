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
def machine_translate(model, tokenizer, texts: int, src_lang, tgt_lang, device="cpu"):  
    tokenizer.src_lang = src_lang
    translated_texts = []
    for text in texts:
        encoded_src = tokenizer(texts, return_tensors="pt").to(device)
        generated_tokens = model.generate(**encoded_src, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
        out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_texts.append(out[0])
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
        file_name = fix_names(name)

        return Image.open(folder_name + file_name)

    else:
        image_list = []

        for file_name_ in file_name:
            # get the name in the right format
            file_name_ = fix_names(file_name_)

            image_list.append(Image.open(folder_name + file_name_))

        return image_list


if __name__ == '__main__':
    # Find GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate the model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # preprocessor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # MT Model and Processor
    MT_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device)
    MT_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    # MT_tokenizer = M2M100Tokenizer.from_pretrained("xlm-mlm-100-1280")

    # set the folder name
    folder_name = 'ms-coco-es/images/'

    # read the dataset
    a = pd.read_excel('ms-coco-es/data/train_human_spanish.xlsx')
    # a = pd.read_excel('ms-coco-es/data/train_machine_english.xlsx')

    # get rid of rows that don't have 5 captions. only 1 has fewer than 5
    a = a.groupby('image_id').filter(lambda x: len(x) >= 5)

    # get the unique image_id in the dataset
    image_id = a['image_id'].unique()

    # get the name
    correct = 0
    correct_k = 0
    total = len(image_id) # run 1000 iterations
    # number of negative samples. In case multiple_images == True, memory requirement could increase as batch size increases
    k = 9
    print("No. of Samples: ", k+1)

    # in case one wants to do multiple inages and one text comparison
    multiple_images = True
    
    # whether translate or directly feed into CLIP
    translate = True
    src_lang = 'es' # Spanish
    tgt_lang = 'en' # English

    for i in range(total):

        name = image_id[i]

        # get the negative samples
        negative_samples = sample_k(image_id, 0, k=k)

        negative_samples = get_captions(a, negative_samples)
        positive_samples = get_captions(a, [name])

        # process the image names
        if multiple_images:
            # in case of multiple images, we append image_id from negative samples
            names = [name] + negative_samples['image_id'].tolist()
            # don't append the captions from negative samples
            text = positive_samples['caption'].tolist()

        else:
            # opposite logic as above
            names = [name]
            
            text = positive_samples['caption'].tolist() + negative_samples['caption'].tolist()
        
        # get the images opened
        image = open_images(folder_name, names)
        
        # translate the text
        if translate:
            text = machine_translate(MT_model, MT_tokenizer, text, src_lang=src_lang, tgt_lang =tgt_lang, device=device)

        # Inference from CLIP
        temp = return_topk_rank(model, processor, image, text, k=5, multiple_images=multiple_images, device=device)
        
        probs, text, topk = temp

        # NDCG
        rank = np.argwhere(topk.indices.squeeze().cpu().numpy() == 0).squeeze()+1 if 0 in temp[2].indices.squeeze() else None
        
        num_candidates = len(temp[2].indices.squeeze())
        # print(temp)
        print(f"[{i} / {total}]: image_name: {name}")
        print(f"Rank of Correct Image: {rank} / {num_candidates}\n")
        

        # top k match
        if 0 in topk.indices.squeeze():
            correct_k += 1

        # top 1 match. We know for sure that the first caption is the ground truth.
        if topk.indices.squeeze()[0] == 0:
            correct += 1
        
        # To avoid OOM error
        del image
        del text
        del temp
        del probs
        del topk
        gc.collect()
        torch.cuda.empty_cache()
    # 0.847 with spanish.
    # 0.974 with english.

    # post-transation: 
    # Top1: 0.906
    # Topk: 1.0
    print("Top 1 Accuracy: ", correct / total)
    print(f"Top {num_candidates} Accuracy: ",correct_k / total)