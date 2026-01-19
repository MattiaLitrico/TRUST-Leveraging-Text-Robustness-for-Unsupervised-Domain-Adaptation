#!/bin/bash
# Training script for GeoNet dataset

# Default data root
DATA_ROOT=${1:-"data/GeoNet"}

# Source and target domains
SOURCE_DOMAIN=${2:-"usa"}
TARGET_DOMAIN=${3:-"asia"}

# Dataset type (GeoPlaces or GeoImNet)
DATASET=${4:-"GeoPlaces"}

# Set number of classes based on dataset
if [ "$DATASET" == "GeoPlaces" ]; then
    NUM_CLASSES=205
elif [ "$DATASET" == "GeoImNet" ]; then
    NUM_CLASSES=600
else
    echo "Unknown dataset: $DATASET. Use 'GeoPlaces' or 'GeoImNet'"
    exit 1
fi

echo "=================================================="
echo "TRUST Training on GeoNet"
echo "Dataset: $DATASET"
echo "Source Domain: $SOURCE_DOMAIN"
echo "Target Domain: $TARGET_DOMAIN"
echo "Number of Classes: $NUM_CLASSES"
echo "=================================================="

# Step 1: Fine-tune BERT on source domain
echo "Step 1: Fine-tuning BERT on source domain..."
python finetune_bert.py \
    --root_dir ${DATA_ROOT}/${DATASET} \
    --metadata_file metadata.json \
    --source_domain ${SOURCE_DOMAIN} \
    --target_domain ${TARGET_DOMAIN} \
    --num_epochs 20 \
    --batch_size 64 \
    --lr 5e-5 \
    --run_name ${DATASET,,}_finetune_bert

# Step 2: Train TRUST
echo "Step 2: Training TRUST..."
python train.py \
    --root_dir ${DATA_ROOT}/${DATASET} \
    --metadata_file metadata.json \
    --source_domain ${SOURCE_DOMAIN} \
    --target_domain ${TARGET_DOMAIN} \
    --num_classes ${NUM_CLASSES} \
    --num_epochs 30 \
    --batch_size 32 \
    --run_name ${DATASET,,}_trust_${SOURCE_DOMAIN}2${TARGET_DOMAIN} \
    --wandb

echo "Training complete!"
