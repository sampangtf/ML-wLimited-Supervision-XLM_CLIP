import gc

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import (CLIPModel, CLIPProcessor,
                          M2M100ForConditionalGeneration, M2M100Tokenizer)

from utils import (fix_names, get_captions, machine_translate, open_images,
                   return_topk_rank, sample_k)
from trainer_fx import MSCOCOES_loader, collate_fn, predict_label

class evaluation_weakbaseline():
    def __init__(self, 
                data_dir, 
                clip_model, 
                negative_sample_k=9, 
                translate=False,
                src_lang:str=None,
                tgt_lang:str=None,
                cuda=False, 
                batch_size=1):
        self.model = clip_model
        self.cuda = cuda
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.model.to(self.device)
        
        dataloader_args = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        if batch_size != 1:
            raise ValueError("So far the model can only be trained with batch size equal to 1.")

        self.train_dataset = MSCOCOES_loader(data_dir, split='train', negative_sample_k=negative_sample_k)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                        batch_size=batch_size, 
                                                        shuffle=True,
                                                        collate_fn=collate_fn,
                                                        **dataloader_args)

        self.val_dataset = MSCOCOES_loader(data_dir, split='val', negative_sample_k=negative_sample_k)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True,
                                                    collate_fn=collate_fn,
                                                    **dataloader_args)

        self.test_dataset = MSCOCOES_loader(data_dir, split='test', negative_sample_k=negative_sample_k)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True,
                                                    collate_fn=collate_fn,
                                                    **dataloader_args)

        # metrics to be tracked
        self.train_loss_list = []
        self.validation_loss_list = []

        self.train_accuracy_list = []
        self.validation_accuracy_list = []

        self.train_recall5_list = []
        self.validation_recall5_list = []

        self.train_recall10_list = []
        self.validation_recall10_list = []

        self.loss_criterion = nn.CrossEntropyLoss()

        # preprocessor
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.translate = translate
        if self.translate: 
            # MT Model and Processor
            self.MT_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(self.device)
            self.MT_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            # MT_tokenizer = M2M100Tokenizer.from_pretrained("xlm-mlm-100-1280")
            self.src_lang, self.tgt_lang = src_lang, tgt_lang

    def evaluate(self, split = 'test'):
        self.model.eval()
        count_total = 0 
        count_correct = 0
        count_recall5 = 0
        count_recall10 = 0 
        loss = 0

        if split == 'train':
            dataloader = self.train_loader
        elif split == 'val':
            dataloader = self.val_loader
        else:
            dataloader = self.test_loader

        for _, batch in enumerate(dataloader):
            text, images, true_img_id = batch

            if self.translate: 
                text = machine_translate(self.MT_model, self.MT_tokenizer, text, src_lang=self.src_lang, tgt_lang=self.tgt_lang, device=self.device)
    
            clip_inputs = self.clip_processor(text=text, images=images, return_tensors="pt", padding=True).to(self.device)
            output_data = self.model(**clip_inputs)

            logits_per_image = output_data.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=0)  # we can take the softmax to get the label probabilities
            
            probs = probs.unsqueeze(0) # to align with batch_size = 1

            # the true image will always be 0th index
            true_img_idx = torch.zeros((1,1)).long().to(self.device)
            curr_loss = self.loss_criterion(probs, true_img_idx)
            
            count_total += len(text) 
            loss += curr_loss
            predicted_labels, top5, top10 = predict_label(probs)
            count_correct += torch.sum(predicted_labels == true_img_idx).cpu().item()
            count_recall5 += torch.sum(torch.isin(top5, true_img_idx)).cpu().item()
            count_recall10 += torch.sum(torch.isin(top10, true_img_idx)).cpu().item()
            

            del text, images, true_img_id, batch
            gc.collect()
            if self.cuda: torch.cuda.empty_cache()

        loss = loss.cpu().detach().numpy()/float(count_total)
        accuracy = float(count_correct)/float(count_total)
        recall5 = float(count_recall5)/float(count_total)
        recall10 = float(count_recall10)/float(count_total)
        return loss, accuracy, recall5, recall10



# class WeakBaseline(nn.Module):
#     def __init__(self, 
#                 translate=True, 
#                 src_lang:str=None,
#                 tgt_lang:str=None,
#                 is_cuda=False):
#         super().__init__()
#         # instantiate the model
#         self.device= torch.device("cuda") if is_cuda else torch.device("cpu")
#         self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
#         self.translate = translate
#         if self.translate: 
#             # MT Model and Processor
#             self.MT_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(self.device)
#             self.MT_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
#             # MT_tokenizer = M2M100Tokenizer.from_pretrained("xlm-mlm-100-1280")
#             self.src_lang, self.tgt_lang = src_lang, tgt_lang
    
#     def forward(self, clip_inputs):
#         ''' params: 
#                 k : number of negative samples (total samples = k + 1)
#         '''
        
#         if self.translate:
#             text = machine_translate(self.MT_model, self.MT_tokenizer, text, src_lang=self.src_lang, tgt_lang=self.tgt_lang, device=self.device)
    
#         output_data = self.model(**clip_inputs)

#         return output_data
