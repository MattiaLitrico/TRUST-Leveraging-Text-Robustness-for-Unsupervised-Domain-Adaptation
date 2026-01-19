import torch
import torch.nn as nn
from dataset import GeoNetDataset, DomainNetDataset
import argparse
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
import time
import wandb
from itertools import zip_longest
from transformers import ViTImageProcessor, ViTForImageClassification, AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from transformers import get_cosine_schedule_with_warmup
import pdb
from utils import *
from models import ViTBase, ClipModel, BERT, ClipBERT, SwinBase
import torch.nn.functional as F
from moco import MoCo
from copy import deepcopy
from supcontrast import SupConLoss
from transformers import AutoTokenizer, AutoProcessor, CLIPModel
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--root_dir', type=str)
parser.add_argument('--metadata_file', type=str)
parser.add_argument('--source_domain', type=str)
parser.add_argument('--target_domain', type=str)
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--num_neighbors', default=10, type=int)
parser.add_argument('--num_classes', default=600, type=int)

parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)

parser.add_argument('--run_name', type=str)
parser.add_argument('--wandb', action='store_true', help="Use wandb")
parser.add_argument('--resume', action='store_true', help="Resume training")

args = parser.parse_args()

# Initialize wandb
if args.wandb:
    wandb.init(project="GeoAdapt", name = args.run_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpuid)

def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

def compute_neighbors_accuracy(idx_t, features, features_text, features_img_bank, probs_img_bank, features_text_bank, probs_text_bank, banks):
    neighbors_img = get_distances(features, features_img_bank).sort()[1][:,:10]
    neighbors_text = get_distances(features_text, features_text_bank).sort()[1][:,:10]
    gt = banks['gt_labels']
    
    gt_neighbors_img = gt[neighbors_img]
    gt_neighbors_text = gt[neighbors_text]
    pdb.set_trace()

@torch.no_grad()
def soft_k_nearest_neighbors(idx_t, features, features_bank, probs_bank):
    pred_probs = []

    for feats in features.split(64):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.num_neighbors]
        
        # (64, num_nbrs, num_classes), average over dim=1
        
        #Compute text similarity
        features = torch.nn.functional.normalize(features, dim=1)
        features_bank = torch.nn.functional.normalize(features_bank, dim=1)
        temperature_scaling = 0.1
        sim_weights = (torch.mm(features.float(), features_bank.transpose(0,1).float()).gather(1, idxs)/temperature_scaling).softmax(dim=1)

        #probs = probs_bank[idxs, :].mean(1)
        probs = (probs_bank[idxs, :] * sim_weights.unsqueeze(2)).sum(1)
        
        pred_probs.append(probs)
        
    pred_probs = torch.cat(pred_probs)
    
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs


def refine_predictions(
    idx_t, features, features_bank, probs_bank):
    pred_labels, probs = soft_k_nearest_neighbors(idx_t, features, features_bank, probs_bank)

    return pred_labels, probs

def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    # in class_aware mode, do not contrast with same-class samples
    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss

# Training function
def train(vit, bert, clip, banks, train_source_loader, train_target_loader, optimizer, criterion, ctr_criterion, epoch):
    vit.train()
    bert.train()
    total_loss = 0
    total_loss_ctr = 0
    pseudo_labels_accuracy = 0

    train_source_iter = iter(train_source_loader)
    train_target_iter = iter(train_target_loader)

    for i, (batch_s, batch_t) in enumerate(zip_longest(train_source_iter, train_target_iter)):
        if batch_s is None:
            train_source_iter = iter(train_source_loader)
            batch_s = next(train_source_iter)
        if batch_t is None:
            train_target_iter = iter(train_target_loader)
            batch_t = next(train_target_iter)
        
        images_s = batch_s[0].to(device)
        strong_images_s = batch_s[5].to(device)
        label_s = batch_s[1].to(device)
        text_s = batch_s[2]

        images_t = batch_t[0].to(device)
        strong_images_t = batch_t[5].to(device)
        label_t = batch_t[1].to(device)
        text_t = batch_t[2]
        idx_t = batch_t[3]

        clip_images_t = batch_t[6]
        clip_text_t = batch_t[4]
        
        optimizer.zero_grad()
        
        outputs_s, loss_s, feats_s = vit(images_s, labels=label_s, return_feats=True)
        _, _, strong_feats_s = vit(strong_images_s, labels=label_s, return_feats=True)
        with torch.no_grad():
            pseudo_label_s, _, feats_s_text = bert(text_s, label_s, return_feats=True)
        
        with torch.no_grad():
            logits_t_text, feats_t_text = bert(text_t, None, return_feats=True)
            pseudo_label_t = logits_t_text.max(1)[1]

        _, clip_weights = clip(image=clip_images_t, text=clip_text_t, labels=pseudo_label_t, max_imgtext_sim=max_cls_imgtext_sims, clip_only=True)
        
        outputs_t, loss_t, feats_t = vit(images_t, labels=pseudo_label_t, return_feats=True, weights=clip_weights)
        _, loss_t_strong, strong_feats_t = vit(strong_images_t, labels=outputs_t.max(1)[1], return_feats=True, weights=(1-clip_weights))
        

        with torch.no_grad():
            ctr_feats_s = torch.stack((feats_s, strong_feats_s), dim=1)
            ctr_feats_t = torch.stack((feats_t, strong_feats_t), dim=1)
            feats_ctr = torch.nn.functional.normalize(torch.cat((ctr_feats_s, ctr_feats_t), dim=0), dim=2)
            feats_ctr_text = torch.nn.functional.normalize(torch.cat((feats_s_text, feats_t_text), dim=0), dim=1)
            
            text_similarity = torch.mm(feats_ctr_text.float(), feats_ctr_text.transpose(0,1).float())
            
        loss_ctr = ctr_criterion(feats_ctr, torch.cat((label_s, pseudo_label_t), dim=0), similarity=text_similarity)
        pseudo_labels_accuracy += (logits_t_text.max(1)[1] == label_t).sum().item()      
        
        loss = loss_s + loss_t + loss_t_strong + loss_ctr
 
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_loss_ctr += 0#(loss_ctr_s + loss_ctr_t).item()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i}/{max(len(train_source_loader),len(train_target_loader))}], Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / max(len(train_source_loader),len(train_target_loader))
    avg_train_loss_ctr = total_loss_ctr / max(len(train_source_loader),len(train_target_loader))
    pseudo_labels_accuracy = pseudo_labels_accuracy / max(len(train_source_loader.dataset),len(train_target_loader.dataset)) * 100.0

    print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_train_loss:.4f}, Loss_ctr: {avg_train_loss_ctr:.4f}")

    return avg_train_loss, pseudo_labels_accuracy, pseudo_labels_refined_accuracy, pseudo_labels_img_refined_accuracy

# Evaluation function
@torch.no_grad()
def evaluate(vit, bert, val_source_loader, val_target_loader):
    vit.eval()
    bert.eval()

    val_target_loss = 0
    val_target_accuracy = 0

    features_img, logits_img, features_text, logits_text, gt_labels, idxs = [], [], [], [], [], []
    
    for batch_t in val_target_loader:
        images_t, labels_t, text_t, idx_t = batch_t[0].to(device), batch_t[1].to(device), batch_t[2], batch_t[3].to(device)

        # Forward pass for target domain
        outputs_t, loss_t, feats_t = vit(images_t, labels=labels_t, return_feats=True)
        _, predicted_t = torch.max(outputs_t, 1)
        features_img.append(feats_t)
        logits_img.append(outputs_t)
        gt_labels.append(labels_t)
        idxs.append(idx_t)

        outputs_t_text, loss_t_text, feats_t_text = bert(text_t, labels=labels_t, return_feats=True)
        features_text.append(feats_t_text)
        logits_text.append(outputs_t_text)

        # Accumulate loss and correct predictions
        val_target_loss += loss_t.item()
        val_target_accuracy += (predicted_t == labels_t).sum().item()
        

    features_img = torch.cat(features_img)
    logits_img = torch.cat(logits_img)
    features_text = torch.cat(features_text)
    logits_text = torch.cat(logits_text)
    gt_labels = torch.cat(gt_labels)
    idxs = torch.cat(idxs)

    probs_img = F.softmax(logits_img, dim=1)
    probs_text = F.softmax(logits_text, dim=1)
    rand_idxs = torch.randperm(len(features_img)).cuda()
    
    banks = {
        "features_img": features_img[rand_idxs][: 16384],
        "features_text": features_text[rand_idxs][: 16384],
        "probs_img": probs_img[rand_idxs][: 16384],
        "probs_text": probs_text[rand_idxs][: 16384],
        "gt_labels": gt_labels[rand_idxs][: 16384],
        "idxs": idxs[rand_idxs][: 16384]
    }
    
    avg_val_target_loss = val_target_loss / len(val_target_loader)
    avg_val_target_accuracy = val_target_accuracy / len(val_target_loader.dataset) * 100

    return banks, avg_val_target_loss, avg_val_target_accuracy

@torch.no_grad()
def compute_imgtext_sims(clip, train_source_loader):
    clip.eval()
   
    imgtext_sims = torch.zeros(len(train_source_loader.dataset)).to(device)
    labels = torch.zeros(len(train_source_loader.dataset)).long().to(device)
    max_cls_imgtext_sims = []
    with torch.no_grad():
        for batch in train_source_loader:
            clip_images_s = batch[6]
            clip_text_s = batch[4]
            label_s = batch[1].to(device)
            idx_s = batch[3]

            clip_weights, _ = clip(image=clip_images_s, text=clip_text_s, labels=None, max_imgtext_sim=None, clip_only=True)
            #imgtext_sims.append(imgtext_sim)
            imgtext_sims[idx_s] = clip_weights
            labels[idx_s] = label_s
    
    for i in range(args.num_classes):
        max_cls_imgtext_sims.append(imgtext_sims[labels == i].max())
    
    return torch.tensor(max_cls_imgtext_sims).to(device)


# Instantiate ViT model
if "DomainNet" in args.root_dir:
    vit = SwinBase(num_classes=args.num_classes).to(device)
elif "GeoNet" in args.root_dir:
    vit = ViTBase(num_classes=args.num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
ctr_criterion = SupConLoss()

# Instantiate BERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=args.num_classes).to(device)
# Load trained model weights
bert_model.load_state_dict(torch.load("logs/"+args.run_name.split("_")[0]+"_finetune_bert/"+args.source_domain+"/"+args.run_name.split("_")[0]+"_finetune_bert_best.tar")['weights'])
bert = BERT(bert_model, tokenizer)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)    
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

if args.resume:
    #vit.load_state_dict(torch.load("logs/GeoAdapt/"+args.source_domain+"2"+args.target_domain+"/GeoAdapt_best.tar")['weights'])
    vit.load_state_dict(torch.load("logs/"+args.run_name+"/vit_best.tar")['weights'])
    #bert.load_state_dict(torch.load("logs/"+args.run_name+"/bert_best.tar")['weights'])
#vit.load_state_dict(torch.load("logs/finetune_vit_old/"+args.source_domain+"/finetune_vit_best.tar")['weights'])

if "DomainNet" in args.root_dir:
    train_source_dataset = DomainNetDataset(args.root_dir, args.metadata_file, args.source_domain, 'train', tokenizer=tokenizer, model='swin', max_length_text=256)
    train_target_dataset = DomainNetDataset(args.root_dir, args.metadata_file, args.target_domain, 'train', tokenizer=tokenizer, model='swin', istarget=True, max_length_text=256)
    val_source_dataset = DomainNetDataset(args.root_dir, args.metadata_file, args.source_domain, 'test', tokenizer=tokenizer, model='swin', max_length_text=256)
    val_target_dataset = DomainNetDataset(args.root_dir, args.metadata_file, args.target_domain, 'test', tokenizer=tokenizer, model='swin', istarget=True, max_length_text=256)
elif "GeoNet" in args.root_dir:
    # Assuming you have your training and validation data ready in the format described above
    train_source_dataset = GeoNetDataset(args.root_dir, args.metadata_file, args.source_domain, 'train', tokenizer=tokenizer, model='vit', max_length_text=256)
    train_target_dataset = GeoNetDataset(args.root_dir, args.metadata_file, args.target_domain, 'train', tokenizer=tokenizer, model='vit', istarget=True, max_length_text=256)
    val_source_dataset = GeoNetDataset(args.root_dir, args.metadata_file, args.source_domain, 'test', tokenizer=tokenizer, model='vit', max_length_text=256)
    val_target_dataset = GeoNetDataset(args.root_dir, args.metadata_file, args.target_domain, 'test', tokenizer=tokenizer, model='vit', istarget=True, max_length_text=256)

train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_source_loader = DataLoader(val_source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
val_target_loader = DataLoader(val_target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

clip = ClipBERT(clip_model, clip_processor, bert_model, tokenizer, class_names=train_target_dataset.class_names)

# Define loss function and optimizer
optimizer = optim.SGD(vit.parameters(), lr=3e-4)

total_steps = max(len(train_source_loader), len(train_target_loader)) * args.num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

best_test_acc = 0

banks, _, _ = evaluate(vit, bert, val_source_loader, val_target_loader)

max_cls_imgtext_sims = None#compute_imgtext_sims(clip, train_source_loader)

for epoch in range(args.num_epochs):
    # Training loop
    train_loss, pseudo_labels_accuracy, pseudo_labels_refined_accuracy, pseudo_labels_img_refined_accuracy = train(vit, bert, clip, banks, train_source_loader, train_target_loader, optimizer, criterion, ctr_criterion, epoch)
    
    # Evaluate the model
    banks, test_target_loss, test_target_accuracy = evaluate(vit, bert, val_source_loader, val_target_loader)

    if test_target_accuracy > best_test_acc:
        #save_weights(vit, epoch, "logs/GeoAdapt/"+args.source_domain+"2"+args.target_domain+"/GeoAdapt_best.tar")
        save_weights(vit, epoch, "logs/"+args.run_name+"/vit_best.tar")
        #save_weights(bert, epoch, "logs/"+args.run_name+"/bert_best.tar")

    print(f'Training Loss: {train_loss:.4f}')
    print(f'Test Target Loss: {test_target_loss:.4f}')
    print(f'Test Target Accuracy: {test_target_accuracy:.2f}%')

    if args.wandb:
        wandb.log({
            'train/loss': train_loss, \
            'train/pseudo_labels_accuracy': pseudo_labels_accuracy, \
            'train/pseudo_labels_refined_accuracy': pseudo_labels_refined_accuracy, \
            'train/pseudo_labels_img_refined_accuracy': pseudo_labels_img_refined_accuracy, \
            'test/target_loss': test_target_loss, \
            'test/target_acc': test_target_accuracy, \
            }, step=epoch) 
