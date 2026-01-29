#!/bin/bash
# Reproduce SAT Paper Results on Selective Classification
# Settings from Table (Coverage vs Error Rate)
# Paper settings: VGG-16, 300 epochs, SAT momentum 0.99, LR decay every 25 epochs

ARCH=vgg16_bn
LOSS=sat
DATASET=cifar10
PRETRAIN=0  # Start SAT immediately (start epoch = 0)
EPOCHS=300
SAT_MOM=0.99  # Paper uses 0.99 for selective classification
SAVE_DIR='./checkpoints/sat_paper_reproduction'
GPU_ID=0

mkdir -p ./checkpoints

echo "========================================="
echo "Reproducing SAT Paper Results"
echo "Dataset: ${DATASET}"
echo "Architecture: ${ARCH}"
echo "Loss: ${LOSS}"
echo "Epochs: ${EPOCHS}"
echo "SAT Momentum: ${SAT_MOM}"
echo "Pretrain: ${PRETRAIN}"
echo "========================================="

### train
python3 -u train.py \
    --arch ${ARCH} \
    --gpu-id ${GPU_ID} \
    --pretrain ${PRETRAIN} \
    --epochs ${EPOCHS} \
    --sat-momentum ${SAT_MOM} \
    --loss ${LOSS} \
    --dataset ${DATASET} \
    --save ${SAVE_DIR} \
    2>&1 | tee ${SAVE_DIR}_training.log

### eval
python3 -u train.py \
    --arch ${ARCH} \
    --gpu-id ${GPU_ID} \
    --epochs ${EPOCHS} \
    --loss ${LOSS} \
    --dataset ${DATASET} \
    --save ${SAVE_DIR} \
    --evaluate \
    2>&1 | tee ${SAVE_DIR}_eval.log

echo "========================================="
echo "Training and evaluation complete!"
echo "Check results in: ${SAVE_DIR}_eval.log"
echo "========================================="
