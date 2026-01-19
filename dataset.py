import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pdb
import json
from os.path import join
import numpy as np
from utils import *
import sys
from rake_nltk import Rake
import random

class GeoNetDataset(Dataset):
    def __init__(self, root_dir, metadata_file, domain, mode, tokenizer, model, max_length_text=160, keywords=False, use_generated_captions=False, istarget=False):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(join(root_dir, domain, mode)))
        self.model = model
        self.max_length_text = max_length_text
        self.keywords = keywords
        self.use_generated_captions = use_generated_captions
        self.istarget = istarget
        self.mode = mode
        if self.keywords:
            language = "en"
            max_ngram_size = 3
            deduplication_threshold = 0.3
            numOfKeywords = 10
            self.custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
        
        with open(join(self.root_dir, metadata_file), 'r') as metadata_file:
           metadata = json.load(metadata_file)[domain+"_"+mode]
        
        self.images = []
        self.labels = []
        self.class_names = []
        self.text = []
        self.locations = []
        
        for i in range(len(metadata['images'])):
            annotations_idx = self.find_index(metadata['annotations'], metadata['images'][i]['id'])
            metadata_idx = self.find_index(metadata['metadata'], metadata['images'][i]['id'])
            
            if metadata_idx == -1 or annotations_idx == -1:
                continue
            
            self.images.append(join(root_dir, metadata['images'][i]['filename']))
            
            if metadata['images'][i]['id'] != metadata['annotations'][annotations_idx]['image_id']:
                print("something wrong!")
                continue
            if metadata['images'][i]['id'] != metadata['metadata'][metadata_idx]['image_id']:
                print("something wrong!")
                continue

            self.labels.append(metadata['annotations'][annotations_idx]['category'])
            
            self.class_names.append(metadata['images'][i]['filename'].split("/")[2])
            
            caption = metadata['metadata'][metadata_idx]['caption']
            description = metadata['metadata'][metadata_idx]['description']
            tags = metadata['metadata'][metadata_idx]['tags']

            text = caption + "," + description + "," + tags
            self.text.append(text)
        
        print("Dataset ", mode, " size: ", len(self.images))

        np.random.seed(1234)   
        dataset_length = len(self.images)
        self.rand_idxs = np.random.permutation(np.arange(dataset_length))
        self.images = np.array(self.images)[self.rand_idxs][:int(dataset_length*1)]
        self.labels = np.array(self.labels)[self.rand_idxs][:int(dataset_length*1)]
        self.class_names = np.array(self.class_names)[self.rand_idxs][:int(dataset_length*1)]
        self.text = np.array(self.text)[self.rand_idxs][:int(dataset_length*1)]

        self.class_names = np.unique(self.class_names[np.argsort(self.labels)])
        
        if istarget:
            if domain == "usa":
                pseudolabels_file = open("logs/finetune_bert/geoplaces_asia_usa.txt", "r")
            else:
                pseudolabels_file = open("logs/finetune_bert/geoplaces_usa_asia.txt", "r")
        
        if istarget and mode == 'train':
            pseudo_labels = {"image":[], "label":[]}

            for line in pseudolabels_file:
                pseudo_labels["image"].append(line.split(" ")[0])
                pseudo_labels["label"].append(int(line.split(" ")[1]))
            
            self.pseudo_labels = pseudo_labels

        if self.model == 'vit':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize input images
                transforms.ToTensor(),           # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image data
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),  # Resize input images
                transforms.ToTensor(),           # Convert images to PyTorch tensors
            ])

        self.strong_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.transform_clip = transforms.Compose([
            transforms.Resize((224, 224)), 
        ]) 

        self.tokenizer = tokenizer

    def find_index(self, metadata, image_id):
        for i in range(len(metadata)):
            if metadata[i]['image_id'] == image_id:
                return i
        return -1

    def __len__(self):
        return len(self.images)

    def get_class_names(self, class_idx):
        #idx = self.labels[self.labels == class_idx.item()][0]
        return self.class_names[class_idx]

    def __getitem__(self, idx):
        image_path = self.images[idx]
        if self.istarget and self.mode == 'train':
            pseudolabel = self.pseudo_labels['label'][self.pseudo_labels['image'].index(os.path.splitext(os.path.basename(image_path))[0])]

        label = self.labels[idx]
        text = self.text[idx] #if self.use_generated_captions == False else open(image_path.replace("jpg", "txt"), "r").read()
        
        image = Image.open(image_path).convert("RGB")

        if self.transform_clip:
            image_clip = self.transform_clip(image)

        if self.strong_transform:
            image_strong = self.strong_transform(image)

        if self.transform:
            image = self.transform(image)

        image_clip = torch.as_tensor(np.array(image_clip))

        #if self.keywords:
        #    text, keywords = extract_keywords(text, self.custom_kw_extractor)
        
        inputs_text = text_to_inputs(text, self.tokenizer, self.max_length_text) if self.tokenizer is not None else text
        
        if self.istarget and self.mode == 'train':
            return [image, label, inputs_text, idx, pre_process_text(text), image_strong, image_clip, image_path, pseudolabel]
        else:
            return [image, label, inputs_text, idx, pre_process_text(text), image_strong, image_clip, image_path]


class DomainNetDataset(Dataset):
    def __init__(self, root_dir, metadata_file, domain, mode, tokenizer, model, max_length_text=160, keywords=False, use_generated_captions=False, istarget=False):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(join(root_dir, domain)))
        self.model = model
        self.max_length_text = max_length_text
        self.keywords = keywords
        self.use_generated_captions = use_generated_captions
        self.istarget = istarget
        self.mode = mode
        if self.keywords:
            language = "en"
            max_ngram_size = 3
            deduplication_threshold = 0.3
            numOfKeywords = 10
            self.custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
        
        with open(join(self.root_dir, metadata_file), 'r') as metadata_file:
           metadata = json.load(metadata_file)[domain+"_"+mode]
        
        self.images = []
        self.labels = []
        self.class_names = []
        self.text = []
        self.locations = []
        
        for i in range(len(metadata['images'])):
            self.images.append(join(root_dir, metadata['images'][i]['filename']))
            
            if metadata['images'][i]['id'] != metadata['annotations'][i]['image_id']:
                print("something wrong!")
                continue
            if metadata['images'][i]['id'] != metadata['metadata'][i]['image_id']:
                print("something wrong!")
                continue

            self.labels.append(metadata['annotations'][i]['category'])
            
            self.class_names.append(metadata['annotations'][i]['class_name'])
            
            text = metadata['metadata'][i]['blip2_cap']
            self.text.append(text)
        
        print("Dataset ", mode, " size: ", len(self.images))

        np.random.seed(1234)   
        dataset_length = len(self.images)
        self.rand_idxs = np.random.permutation(np.arange(dataset_length))
        self.images = np.array(self.images)[self.rand_idxs][:int(dataset_length*1)]
        self.labels = np.array(self.labels)[self.rand_idxs][:int(dataset_length*1)]
        self.class_names = np.array(self.class_names)[self.rand_idxs][:int(dataset_length*1)]
        self.text = np.array(self.text)[self.rand_idxs][:int(dataset_length*1)]

        self.class_names = np.unique(self.class_names[np.argsort(self.labels)])
        
        if self.model == 'vit' or self.model == 'swin':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize input images
                transforms.ToTensor(),           # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image data
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),  # Resize input images
                transforms.ToTensor(),           # Convert images to PyTorch tensors
            ])

        self.strong_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.transform_clip = transforms.Compose([
            transforms.Resize((224, 224)), 
        ]) 

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.images)

    def get_class_names(self, class_idx):
        #idx = self.labels[self.labels == class_idx.item()][0]
        return self.class_names[class_idx]

    def __getitem__(self, idx):
        image_path = self.images[idx]

        label = self.labels[idx]
        text = self.text[idx] #if self.use_generated_captions == False else open(image_path.replace("jpg", "txt"), "r").read()
        
        image = Image.open(image_path).convert("RGB")

        if self.transform_clip:
            image_clip = self.transform_clip(image)

        if self.strong_transform:
            image_strong = self.strong_transform(image)

        if self.transform:
            image = self.transform(image)

        image_clip = torch.as_tensor(np.array(image_clip))

        #if self.keywords:
        #    text, keywords = extract_keywords(text, self.custom_kw_extractor)
        
        inputs_text = text_to_inputs(text, self.tokenizer, self.max_length_text) if self.tokenizer is not None else text
        
        return [image, label, inputs_text, idx, pre_process_text(text), image_strong, image_clip, image_path]

