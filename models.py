from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor, SwinForImageClassification
from transformers import AutoTokenizer, AutoProcessor, CLIPModel, AutoImageProcessor
import torch.nn as nn
import torch
import pdb
from torch.nn import CrossEntropyLoss
from utils import *

# Define ViT model
class ViTBase(nn.Module):
    def __init__(self, num_classes):
        super(ViTBase, self).__init__()
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes, ignore_mismatched_sizes=True)
        self.criterion = CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes
        #self.model.classifier = nn.Sequential(nn.Linear(768,256), nn.ReLU(), nn.Linear(256,self.model.classifier.out_features))
        

    def forward(self, img, labels=None, return_feats=False, weights=None):
        
        out = self.model.vit(img)
        
        sequence_output = out[0]
        
        logits = self.model.classifier(sequence_output[:, 0, :])
        if labels is not None:
            if weights is not None:
                loss = (weights * self.criterion(logits.view(-1, self.num_classes), labels.view(-1))).mean()
            else:  
                loss = self.criterion(logits.view(-1, self.num_classes), labels.view(-1)).mean()

        if return_feats:
            feats = sequence_output[:,0,:]#sequence_output[:,1:,:].mean(dim=1)
            
            if labels is None:
                return logits, feats
            return logits, loss, feats  

        if labels is None:
                return logits
        return logits, loss

# Define Swin model
class SwinBase(nn.Module):
    def __init__(self, num_classes):
        super(SwinBase, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
        self.model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
        self.criterion = CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes
        self.model.classifier = nn.Sequential(nn.Linear(1024,512), nn.ReLU(), nn.Linear(512,self.num_classes))
        

    def forward(self, img, labels=None, return_feats=False, weights=None):
        out = self.model.swin(img)
        
        sequence_output = out[0]
        
        logits = self.model.classifier(sequence_output[:, 0, :])
        
        if labels is not None:
            if weights is not None:
                loss = (weights * self.criterion(logits.view(-1, self.num_classes), labels.view(-1))).mean()
            else:  
                loss = self.criterion(logits.view(-1, self.num_classes), labels.view(-1)).mean()

        if return_feats:
            feats = sequence_output[:,0,:]#sequence_output[:,1:,:].mean(dim=1)
            
            if labels is None:
                return logits, feats
            return logits, loss, feats  

        if labels is None:
                return logits
        return logits, loss

# Define ViT model
class BERT(nn.Module):
    def __init__(self, model, tokenizer):
        super(BERT, self).__init__()
        self.tokenizer = tokenizer
        self.model = model

        self.criterion = CrossEntropyLoss()
        self.num_classes = self.model.num_labels

    def forward(self, text, labels=None, return_feats=False):
        input_ids = text['ids'].to(self.model.device)
        attention_mask = text['mask'].to(self.model.device)
        #out = self.model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        distilbert_output = self.model.distilbert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_state = distilbert_output[0]  
        pooled_output = hidden_state[:, 0]  
        pooled_output = self.model.pre_classifier(pooled_output)  
        pooled_output = nn.ReLU()(pooled_output)  
        pooled_output = self.model.dropout(pooled_output)  
        logits = self.model.classifier(pooled_output)  
        
        if labels is not None:  
            loss = self.criterion(logits.view(-1, self.num_classes), labels.view(-1))
        
        if return_feats:
            feats = pooled_output#hidden_state[:,1:,:].mean(dim=1)
            if labels is None:
                return logits, feats
            return logits, loss, feats

        if labels is None:
                return logits
        return logits, loss

# Define ViT model
class ClipBERT(nn.Module):
    def __init__(self, clip_model, clip_processor, bert_model, bert_tokenizer, class_names):
        super(ClipBERT, self).__init__()
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        self.criterion = CrossEntropyLoss(reduction='none') 
        self.num_classes = self.bert_model.num_labels

        self.class_text = ["a photo of a {}".format(c) for c in class_names]

    """def forward(self, image, text, labels=None, return_feats=False):
        features = []
        
        for img, text in zip(image, text):
            #Computing weights using CLIP
            keywords = text.split("-")
            
            text = ["a photo of {}".format(k[:77]) for k in keywords]
            
            inputs = self.clip_processor(images=img, text=text, padding=True, return_tensors="pt").to(self.bert_model.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            weights = outputs['logits_per_image'].softmax(1)
            
            input_ids = []
            attention_mask = [] 
            
            text = np.array(keywords)[torch.topk(weights, min(3, weights.shape[1]), dim=1)[1][0].cpu()]
            text = " ".join(text) if len(keywords) > 1 else text
            
            text = self.bert_tokenizer.encode_plus( text, None, add_special_tokens=True, max_length=512, pad_to_max_length=True, return_token_type_ids=True)
            
            #EXTRACTING FEATURES
            #for k in keywords:
                #k = text_to_inputs(k, self.bert_tokenizer)
                
                #k = self.bert_tokenizer.encode_plus( k, None, add_special_tokens=True, max_length=512, pad_to_max_length=True, return_token_type_ids=True)

                #input_ids.append(torch.tensor(k['input_ids'], dtype=torch.long).to(self.bert_model.device))
                #attention_mask.append(torch.tensor(k['attention_mask'], dtype=torch.long).to(self.bert_model.device))
                

            input_ids = torch.tensor(text['input_ids'], dtype=torch.long).to(self.bert_model.device)
            attention_mask = torch.tensor(text['attention_mask'], dtype=torch.long).to(self.bert_model.device)  

            #input_ids = torch.stack(input_ids, dim=0)
            #attention_mask = torch.stack(attention_mask, dim=0)
            
            distilbert_output = self.bert_model.distilbert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            pdb.set_trace()
            hidden_state = distilbert_output[0]  
            feats = hidden_state[:, 0]
            
            #weighted_text_feats = (weights[0].unsqueeze(1) * feats).sum(0)
            #features.append(weighted_text_feats)
            features.append(feats[0])
        
        features = torch.stack(features)
        
        pooled_output = self.bert_model.pre_classifier(features)  
        pooled_output = nn.ReLU()(pooled_output)  
        pooled_output = self.bert_model.dropout(pooled_output)  
        logits = self.bert_model.classifier(pooled_output)  
        
        if labels is not None:  
            loss = self.criterion(logits.view(-1, self.num_classes), labels.view(-1))
        
        if return_feats:
            feats = pooled_output#hidden_state[:,1:,:].mean(dim=1)
            if labels is None:
                return logits, feats
            return logits, loss, feats

        if labels is None:
                return logits
        return logits, loss"""

    def forward(self, image, text, labels=None, return_feats=False, max_imgtext_sim=None, clip_only=False):
        #for img, txt in zip(image, text):
        #text = self.class_text
        inputs = self.clip_processor(images=image, text=[txt[:75] for txt in text], padding=True, truncation=True, return_tensors="pt").to(self.bert_model.device)
        #text1 = self.clip_processor(images=image, text=[txt[:75] for txt in text], padding=True, truncation=True, return_tensors="pt").to(self.bert_model.device)
        #text2 = self.clip_processor(images=image, text=[self.class_text[l][:75] for l in labels], padding=True, truncation=True, return_tensors="pt").to(self.bert_model.device)
        with torch.no_grad():
            #feats1 = self.clip_model(**text1)['text_embeds']
            #feats2 = self.clip_model(**text2)['text_embeds']

            #feats1 = torch.nn.functional.normalize(feats1, dim=1)
            #feats2 = torch.nn.functional.normalize(feats2, dim=1)
            #pdb.set_trace()
            #text_similarity = torch.mm(feats1.float(), feats2.transpose(0,1).float())
            outputs = self.clip_model(**inputs)
            #if clip_only:
                #weights = (outputs.logits_per_image.softmax(dim=1).argmax(1) == labels).int()
                #return torch.diagonal(outputs['logits_per_image'], 0), weights
                
            #    return torch.diagonal(text_similarity, 0), torch.diagonal(text_similarity, 0)

            #weights = torch.zeros(image.shape[0]).to(self.bert_model.device)
            #weights[torch.topk(torch.diagonal(outputs['logits_per_image'], 0), k=int(image.shape[0]*0.8))[1]] = 1
            #Versione solo similarità con rispettiva caption
            
            #if max_imgtext_sim is not None:
            #    weights = torch.diagonal(outputs['logits_per_image'], 0) / max_imgtext_sim[labels]#max_imgtext_sim
            #else:
            #    weights = torch.diagonal(outputs['logits_per_image'], 0) / torch.diagonal(outputs['logits_per_image'], 0).max()
            weights = torch.diagonal(torch.softmax(outputs['logits_per_image']/3.0, dim=1), 0)
        
        if clip_only:
            return torch.diagonal(outputs['logits_per_image'], 0), weights
        
        text = self.bert_tokenizer.batch_encode_plus( [txt[:160] for txt in text], add_special_tokens=True, max_length=512, pad_to_max_length=True, return_token_type_ids=True)

        input_ids = torch.tensor(text['input_ids'], dtype=torch.long).to(self.bert_model.device)
        attention_mask = torch.tensor(text['attention_mask'], dtype=torch.long).to(self.bert_model.device)  
        
        distilbert_output = self.bert_model.distilbert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_state = distilbert_output[0]  
        feats = hidden_state[:, 0]
        
        pooled_output = self.bert_model.pre_classifier(feats)  
        pooled_output = nn.ReLU()(pooled_output)  
        pooled_output = self.bert_model.dropout(pooled_output)  
        logits = self.bert_model.classifier(pooled_output)  
        
        if labels is not None:  
            loss = (weights * self.criterion(logits.view(-1, self.num_classes), labels.view(-1))).mean()
            #loss = (self.criterion(logits.view(-1, self.num_classes), labels.view(-1))).mean()
        
        if return_feats:
            feats = pooled_output#hidden_state[:,1:,:].mean(dim=1)
            if labels is None:
                return logits, feats
            return logits, loss, feats, torch.diagonal(outputs['logits_per_image'], 0)

        if labels is None:
                return logits
        return logits, loss, torch.diagonal(outputs['logits_per_image'], 0)

class ClipModel(nn.Module):
    def __init__(self, num_classes, criterion, device='cuda'):
        super(ClipModel, self).__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)    
        self.img_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.classifier = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, num_classes))
        
        self.criterion = criterion

    def forward(self, img=None, text=None, labels=None):
        inputs_image = self.img_processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs_image)

        ids = text['ids'].to(self.device)
        mask = text['mask'].to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(ids, mask)
        
        features = torch.cat([image_features, text_features], dim=1)
        
        out = self.classifier(features)
        loss = self.criterion(out, labels) if labels != None else None

        return out, loss
