#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

LABEL_FILE=${1:-"EAT/inference/labels.csv"}
EVAL_DIR=${2:-"/hpc_stor03/sjtu_home/wenxi.chen/mydata/audio/AS20K"}
MODEL_DIR=${3:-"EAT"}
CHECKPOINT_DIR=${4:-"/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/EATs/finetuning/as20k_epoch30/checkpoint_last.pt"}
TARGET_LENGTH=${5:-1024}
DEVICE=${6:-"cuda"}
BATCH_SIZE=${7:-32}

python EAT/evaluation/eval.py  \
    --label_file="$LABEL_FILE" \
    --eval_dir="$EVAL_DIR" \
    --model_dir="$MODEL_DIR" \
    --checkpoint_dir="$CHECKPOINT_DIR" \
    --target_length="$TARGET_LENGTH" \
    --device="$DEVICE" \
    --batch_size="$BATCH_SIZE"

# For optimal performance, 1024 is recommended for 10-second audio clips. (128 for 1-second)
# However, you should adjust the target_length parameter based on the duration and characteristics of your specific audio inputs.
# EAT-finetuned could make evaluation well even given truncated audio clips.