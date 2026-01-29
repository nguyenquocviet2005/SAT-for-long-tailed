#!/bin/bash
# Corrected run script matching the paper's exact specifications
# SAT for Selective Classification on CIFAR-10

ARCH=vgg16_bn
LOSS=sat
DATASET=cifar10
PRETRAIN=0          # Start epoch (ε₀) = 0 (SAT from the beginning)
MOM=0.99            # SAT momentum (μ) = 0.99 for selective classification
SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_$1
GPU_ID=1

mkdir -p ./log

### train
python -u train.py --arch ${ARCH} --gpu-id ${GPU_ID} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} \
       --dataset ${DATASET} --save ${SAVE_DIR}  \
       2>&1 | tee -a ${SAVE_DIR}.log

### eval
python -u train.py --arch ${ARCH} --gpu-id ${GPU_ID} \
       --loss ${LOSS} --reward ${REWARD} --dataset ${DATASET} \
       --save ${SAVE_DIR} --evaluate \
       2>&1 | tee -a ${SAVE_DIR}.log
