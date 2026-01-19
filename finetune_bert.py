import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_cosine_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from dataset import GeoNetDataset, DomainNetDataset
import argparse
import wandb
from utils import *
import pdb
from sampler import BalancedSampler

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--root_dir', type=str)
parser.add_argument('--metadata_file', type=str)
parser.add_argument('--source_domain', type=str)
parser.add_argument('--target_domain', type=str)
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--use_generated_captions', action='store_true', help="Use generated_captions")

parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)

parser.add_argument('--run_name', type=str)
parser.add_argument('--wandb', action='store_true', help="Use wandb")
parser.add_argument('--resume', action='store_true', help="Resume training")

args = parser.parse_args()

# Initialize wandb
if args.wandb:
    wandb.init(project="GeoAdapt", name = args.run_name + "_" + args.source_domain)

MAX_LEN = 512

def train(model, tokenizer, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs_text = batch[2]

        input_ids = inputs_text['ids'].to(device)
        attention_mask = inputs_text['mask'].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)

    return avg_train_loss

def evaluate(model, tokenizer, test_loader, epoch):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs_text = batch[2]

            input_ids = inputs_text['ids'].to(device)
            attention_mask = inputs_text['mask'].to(device)
            labels = batch[1].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            val_accuracy += (logits.argmax(axis=1) == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader.dataset) * 100

    return avg_val_loss, avg_val_accuracy
    
if "GeoPlaces" in args.root_dir:
    num_classes = 205     
elif "GeoImNet" in args.root_dir:
    num_classes = 600       
elif "DomainNet" in args.root_dir:
    num_classes = 345 
print("Number of classes = ", num_classes)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes) # Change num_labels based on your classification task

if args.resume:
    model.load_state_dict(torch.load("logs/" + args.run_name + "/" + args.source_domain + "/" + args.run_name + "_best.tar")['weights'])

# Assuming you have your training and validation data ready in the format described above
if "DomainNet" in args.root_dir:
    train_dataset = DomainNetDataset(args.root_dir, args.metadata_file, args.source_domain, 'train', tokenizer=tokenizer, model='bert', keywords=False, use_generated_captions=False, max_length_text=256)
    val_dataset = DomainNetDataset(args.root_dir, args.metadata_file, args.target_domain, 'test', tokenizer=tokenizer, model='bert', keywords=False, use_generated_captions=False, max_length_text=256)
elif "GeoNet" in args.root_dir:
    train_dataset = GeoNetDataset(args.root_dir, args.metadata_file, args.source_domain, 'train', tokenizer=tokenizer, model='bert', keywords=False, use_generated_captions=False, max_length_text=256)
    val_dataset = GeoNetDataset(args.root_dir, args.metadata_file, args.target_domain, 'test', tokenizer=tokenizer, model='bert', keywords=False, use_generated_captions=False, max_length_text=256)

sampler = BalancedSampler(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
num_epochs = args.num_epochs
total_steps = len(train_loader) * num_epochs

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_test_acc = 0

#Training
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')

    train_loss = train(model, tokenizer, train_loader, optimizer, loss_fn, epoch)
    test_loss, test_accuracy = evaluate(model, tokenizer, val_loader, epoch)

    if test_accuracy > best_test_acc:
        save_weights(model, epoch, "logs/" + args.run_name + "/" + args.source_domain + "/" + args.run_name + "_best.tar")
    
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    if args.wandb:
        wandb.log({
            'train/loss': train_loss, \
            'test/loss': test_loss, \
            'test/acc': test_accuracy
            }, step=epoch) 