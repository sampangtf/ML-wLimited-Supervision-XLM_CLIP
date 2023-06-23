import os
import glob
import gc

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import DataLoader, Dataset
import torch.utils
import torchvision
import torchvision.transforms as transforms
from IPython.display import Image
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

from utils import (fix_names, get_captions, machine_translate, open_images,
                   return_topk_rank, sample_k)

from transformers import CLIPProcessor, CLIPModel, XLMRobertaTokenizer, XLMRobertaModel

class MSCOCOES_loader(Dataset):
    def __init__(self,
                root_dir: str, 
                split:str = 'train',
                negative_sample_k: int = 9,
                transform: torchvision.transforms.Compose = None):
        '''Specify the root directory of the datasets and 
        '''
        self.root = os.path.expanduser(root_dir)
        # self.transform = torchvision.transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.transform = torchvision.transforms
        self.spit = split
        self.negative_sample_k = negative_sample_k

        self.train_eng_caption = None 
        self.train_human_esp_caption = None
        self.validation = None
        self.test = None
        self.image_id_list = list()

        # Image Directory
        self.image_folders = os.path.join(self.root, "images")
        
        # Caption Directory
        self.caption_folders = os.path.join(self.root, "data")
        if split == 'train':
            self.train_eng_caption = pd.read_excel(os.path.join(self.caption_folders, "train_machine_english.xlsx"))
            self.train_human_esp_caption = pd.read_excel(os.path.join(self.caption_folders, "train_human_spanish.xlsx"))
            
            # downsample the english dataset to the same size as the spanish dataset
            esp_data_size = len(self.train_human_esp_caption)
            self.train_eng_caption = self.train_eng_caption.iloc[:esp_data_size]

            # For simplicitiy - simply mix the two datasets
            self.caption_df = pd.concat([self.train_eng_caption, self.train_human_esp_caption])

        elif split == 'val':
            self.caption_df = pd.read_excel(os.path.join(self.caption_folders, "validation.xlsx"))
            # self.image_id_list = self.validation['image_id'].unique().tolist()
        elif split == 'test':
            self.caption_df = pd.read_excel(os.path.join(self.caption_folders, "test.xlsx"))
            
        # get rid of rows that don't have 5 captions. only 1-2 images has fewer than 5 across all dataframe
        self.caption_df = self.caption_df.groupby('image_id').filter(lambda x: len(x) >= 5)
        self.image_id_list += self.caption_df['image_id'].unique().tolist()

        # gathering the entire image datasets
        self.img_paths = []
        img_dir = os.path.join(self.image_folders, '*.jpg')
        files = glob.glob(img_dir)
        self.img_paths += [f for f in files]
        

    def __getitem__(self, index):
        '''
        Returns a list of images, if list contains 1 image, returns just an Image Object, not a list
        :param folder_name:
        :param file_name: a list of file names
        :return:
        '''
        text = None
        true_img_id = None

        true_img_id = self.image_id_list[index]

        # getting the negative image indexes, excluding the true image
        negative_samples = sample_k(self.image_id_list, index, self.negative_sample_k)

        negative_samples = get_captions(self.caption_df, negative_samples)
        positive_samples = get_captions(self.caption_df, [true_img_id])

        # in case of multiple images, we append image_id from negative samples
        image_ids_pool = [true_img_id] + negative_samples['image_id'].tolist()
        # don't append the captions from negative samples
        text = positive_samples['caption'].tolist()

        # open images and appending them to a list
        images = open_images(self.image_folders, image_ids_pool)


        return text, images, true_img_id 
        
    def __len__(self):
        return len(self.image_id_list)


def collate_fn(batch):
    '''only for batch_size = 1 '''
    text_batch = []
    images_batch = []
    true_img_id_batch = []

    # for item in batch:
    #     text, images, true_img_id = item
    #     text_batch.append(text)
    #     images_batch.append(images)
    #     true_img_id_batch.append(true_img_id)

    # return text_batch, images_batch, true_img_id_batch

    text, images, true_img_id = batch[0]
    return text, images, true_img_id
    

def predict_label(model_output_probs):
    pred = torch.argmax(model_output_probs, dim = 1)
    top5 = model_output_probs.reshape(1, -1).topk(5).indices.squeeze()
    top10 = model_output_probs.reshape(1, -1).topk(10).indices.squeeze()
    return pred, top5, top10

# def compute_loss(model, model_output, target_labels, noramlize=True):
#     loss_criterion = model.loss_criterion
#     loss = loss_criterion(model_output, target_labels)
#     if noramlize:
#         loss /= len(target_labels)
#     return loss

def get_optimizer(model, configuration):
    optimizer_type = configuration["optimizer_type"]
    learning_rate = configuration["lr"]
    weight_decay = configuration["weight_decay"]
    momentum  = configuration["momentum"] 

    if optimizer_type =='SGD':
      optimizer = f"torch.optim.{optimizer_type}(model.parameters() , lr={learning_rate},weight_decay={weight_decay}, momentum={momentum})"
  
    else:
      optimizer = f"torch.optim.{optimizer_type}(model.parameters() , lr={learning_rate},weight_decay={weight_decay})"
    return eval(optimizer)
    

class Trainer():
    def __init__(self, 
                data_dir,
                model, 
                optimizer, 
                model_dir, 
                negative_sample_k = 9,
                batch_size=1,
                load_from_disk=True,
                checkpoint_name=None,
                cuda=False):
        self.model_dir = model_dir
        self.model = model
        self.cuda = cuda
        # if cuda:
        #     self.model.cuda()
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
        self.optimizer = optimizer
        
        # metrics to be tracked
        self.train_loss_list = []
        self.validation_loss_list = []

        self.train_accuracy_list = []
        self.validation_accuracy_list = []

        self.train_recall5_list = []
        self.validation_recall5_list = []

        self.train_recall10_list = []
        self.validation_recall10_list = []

        # load the model from the disk if it exists
        if os.path.exists(model_dir) and load_from_disk:
            checkpoint_name = 'checkpoint.pt' if checkpoint_name is None else checkpoint_name
            checkpoint = torch.load(os.path.join(self.model_dir, checkpoint_name))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.model.train()
        
        self.loss_criterion = nn.CrossEntropyLoss()
        
        # Loading pre-trained processor
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.xlm_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
        
    def save_model(self, checkpoint_name=None):
        checkpoint_name = 'checkpoint.pt' if checkpoint_name is None else checkpoint_name

        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                    }, os.path.join(self.model_dir, checkpoint_name))

    def train(self, num_epochs):
        self.model.train()
        train_loss, train_acc, train_recall5, train_recall10 = self.evaluate(split='train')
        val_loss, val_acc, val_recall5, val_recall10 = self.evaluate(split='val')

        self.train_loss_list.append(train_loss)
        self.train_accuracy_list.append(train_acc)
        self.train_recall5_list.append(train_recall5)
        self.train_recall10_list.append(train_recall10)

        self.validation_loss_list.append(val_loss)
        self.validation_accuracy_list.append(val_acc)
        self.validation_recall5_list.append(val_recall5)
        self.validation_recall10_list.append(val_recall10)

        print('Epoch:{}, Training Loss:{:.4f}, Validation Loss:{:.4f}'.format(
        0, self.train_loss_list[-1], self.validation_loss_list[-1])
        )

        for epoch_idx in range(num_epochs):
            self.model.train()
            for _, batch in enumerate(self.train_loader):
                text, images, true_img_id = batch

                xlm_inputs = self.xlm_tokenizer(text, return_tensors="pt").to(self.device)
                clip_inputs = self.clip_processor(text=text, images=images, return_tensors="pt", padding=True).to(self.device)
                output_data = self.model(xlm_inputs=xlm_inputs, **clip_inputs)

                logits_per_image = output_data.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=0)  # we can take the softmax to get the label probabilities
                
                probs = probs.unsqueeze(0) # to align with batch_size = 1
                # the true image will always be 0th index
                true_img_idx = torch.zeros((1,1)).long().to(self.device)
                loss = self.loss_criterion(probs, true_img_idx)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                gc.collect()
                if self.cuda: torch.cuda.empty_cache()
            # Storing the metrics
            train_loss, train_acc, train_recall5, train_recall10 = self.evaluate(split='train')
            val_loss, val_acc, val_recall5, val_recall10 = self.evaluate(split='val')

            self.train_loss_list.append(train_loss)
            self.train_accuracy_list.append(train_acc)
            self.train_recall5_list.append(train_recall5)
            self.train_recall10_list.append(train_recall10)

            self.validation_loss_list.append(val_loss)
            self.validation_accuracy_list.append(val_acc)
            self.validation_recall5_list.append(val_recall5)
            self.validation_recall10_list.append(val_recall10)

            print('Epoch:{} | Training Loss:{:.4f}, Validation Loss:{:.4f} | Training Recall@5:{:.4f}, Validation Recall@5:{:.4f}'.format(
                epoch_idx+1, self.train_loss_list[-1], self.validation_loss_list[-1], self.train_recall5_list[-1], self.validation_recall5_list[-1])
            )
        
            if epoch_idx >0 and epoch_idx % 5 == 0 :
                self.save_model(f"checkpoint_{epoch_idx}.pt")

        self.save_model()

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
            
            xlm_inputs = self.xlm_tokenizer(text, return_tensors="pt").to(self.device)
            clip_inputs = self.clip_processor(text=text, images=images, return_tensors="pt", padding=True).to(self.device)
            output_data = self.model(xlm_inputs=xlm_inputs, **clip_inputs)

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
            
            gc.collect()
            if self.cuda: torch.cuda.empty_cache()

        self.model.train()
        loss = loss.cpu().detach().numpy()/float(count_total)
        accuracy = float(count_correct)/float(count_total)
        recall5 = float(count_recall5)/float(count_total)
        recall10 = float(count_recall10)/float(count_total)
        return loss, accuracy, recall5, recall10

    def plot_loss(self):
        plt.figure()
        count_epoch = range(len(self.train_loss_list))
        
        plt.plot(count_epoch, self.train_loss_list, '-r', label = 'training')
        plt.plot(count_epoch, self.validation_loss_list, '-g', label = 'validation')
        plt.title("Loss Plot against Epochs")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.show()
    
    def plot_accuracy(self):
        plt.figure()
        count_epoch = range(len(self.train_accuracy_list))
        
        plt.plot(count_epoch, self.train_accuracy_list, '-r', label = 'training')
        plt.plot(count_epoch, self.validation_accuracy_list, '-g', label = 'validation')
        plt.title("Accuracy Plot against Epochs")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.show()

    def plot_recall5(self):
        plt.figure()
        count_epoch = range(len(self.train_recall5_list))
        
        plt.plot(count_epoch, self.train_recall5_list, '-r', label = 'training')
        plt.plot(count_epoch, self.validation_recall5_list, '-g', label = 'validation')
        plt.title("Recall@5 Plot against Epochs")
        plt.legend()
        plt.ylabel("Recall@5")
        plt.xlabel("Epochs")
        plt.show()

    def plot_recall10(self):
        plt.figure()
        count_epoch = range(len(self.train_recall10_list))
        
        plt.plot(count_epoch, self.train_recall10_list, '-r', label = 'training')
        plt.plot(count_epoch, self.validation_recall10_list, '-g', label = 'validation')
        plt.title("Recall@10 Plot against Epochs")
        plt.legend()
        plt.ylabel("Recall@10")
        plt.xlabel("Epochs")
        plt.show()
    
    # def final_evaluate(self):
    #     self.model.eval()
        
    #     y_true = []
    #     y_predicted = [] 
        
    #     for _, batch in enumerate(self.test_loader):
    #         if self.cuda:
    #             input_data, target_data = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
    #         else:
    #             input_data, target_data = Variable(batch[0]), Variable(batch[1])
                
    #         output_data = self.model(input_data)
    #         predicted_labels = predict_label(output_data)

    #         if self.cuda:
    #             y_true += list(target_data.cpu())
    #             y_predicted += list(predicted_labels.cpu())
    #         else:
    #             y_true += list(target_data)
    #             y_predicted += list(predicted_labels)
        
    #     self.model.train()
       
    #     precision = precision_score(y_true, y_predicted)
    #     recall = recall_score(y_true, y_predicted)
    #     f1 = f1_score(y_true, y_predicted)
    #     print("Confusion Matrix for the Trained Model")
    #     print(confusion_matrix(y_true, y_predicted),'\n')
        
    #     return precision, recall, f1
        
        

        



