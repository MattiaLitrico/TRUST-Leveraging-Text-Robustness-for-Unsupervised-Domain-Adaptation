#!/bin/bash
# Training script for DomainNet dataset

# Default data root
DATA_ROOT=${1:-"data/DomainNet"}

# Source and target domains
SOURCE_DOMAIN=${2:-"real"}
TARGET_DOMAIN=${3:-"clipart"}

NUM_CLASSES=345

echo "=================================================="
echo "TRUST Training on DomainNet-345"
echo "Source Domain: $SOURCE_DOMAIN"
echo "Target Domain: $TARGET_DOMAIN"
echo "Number of Classes: $NUM_CLASSES"
echo "=================================================="

# Step 1: Fine-tune BERT on source domain
echo "Step 1: Fine-tuning BERT on source domain..."
python finetune_bert.py \
    --root_dir ${DATA_ROOT} \
    --metadata_file metadata.json \
    --source_domain ${SOURCE_DOMAIN} \
    --target_domain ${TARGET_DOMAIN} \
    --num_epochs 20 \
    --batch_size 64 \
    --lr 5e-5 \
    --run_name domainnet_finetune_bert

# Step 2: Train TRUST
echo "Step 2: Training TRUST..."
python train.py \
    --root_dir ${DATA_ROOT} \
    --metadata_file metadata.json \
    --source_domain ${SOURCE_DOMAIN} \
    --target_domain ${TARGET_DOMAIN} \
    --num_classes ${NUM_CLASSES} \
    --num_epochs 30 \
    --batch_size 32 \
    --run_name domainnet_trust_${SOURCE_DOMAIN}2${TARGET_DOMAIN} \
    --wandb

echo "Training complete!"
