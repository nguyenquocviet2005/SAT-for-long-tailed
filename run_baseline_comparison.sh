#!/bin/bash
# Train single-model SAT baseline to compare with SAT-Ensemble
# Matching hyperparameters: epochs=200, pretrain=60, arch=vgg16_bn

# Activate SAT-Ensemble venv (has all dependencies)
source ../SAT-Ensemble/venv/bin/activate

ARCH=vgg16_bn
LOSS=sat
DATASET=cifar10
PRETRAIN=60
EPOCHS=200
MOM=0.9
SAVE_DIR='./checkpoints/sat_baseline_200ep'
GPU_ID=0

mkdir -p ./checkpoints

### train
python3 -u train.py \
    --arch ${ARCH} \
    --gpu-id ${GPU_ID} \
    --pretrain ${PRETRAIN} \
    --epochs ${EPOCHS} \
    --sat-momentum ${MOM} \
    --loss ${LOSS} \
    --dataset ${DATASET} \
    --save ${SAVE_DIR} \
    2>&1 | tee -a ${SAVE_DIR}.log

### eval
python3 -u train.py \
    --arch ${ARCH} \
    --gpu-id ${GPU_ID} \
    --epochs ${EPOCHS} \
    --loss ${LOSS} \
    --dataset ${DATASET} \
    --save ${SAVE_DIR} \
    --evaluate \
    2>&1 | tee -a ${SAVE_DIR}_eval.log
