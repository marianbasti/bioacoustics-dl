#!/bin/bash

# Default values
CUDA_DEVICES="0"
BATCH_SIZE=12
SAVE_DIR=""
DATA_DIR="$(pwd)/data/unlabeled"
WORLD_SIZE=1
RESTORE_FILE=""
TARGET_LENGTH=1024
MASK_RATIO=0.75
NUM_UPDATES=400000
LEARNING_RATE=1.5e-4
UPDATE_FREQ=1
PROJECT_DIR="$(pwd)"
EAT_DIR="${PROJECT_DIR}/EAT"
NNODES=1
NODE_RANK=0
NPROC_PER_NODE=4
MASTER_ADDR="localhost"
MASTER_PORT=29500

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --restore_file)
            RESTORE_FILE="$2"
            shift 2
            ;;
        --target_length)
            TARGET_LENGTH="$2"
            shift 2
            ;;
        --mask_ratio)
            MASK_RATIO="$2"
            shift 2
            ;;
        --num_updates)
            NUM_UPDATES="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --update_freq)
            UPDATE_FREQ="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$SAVE_DIR" ]; then
    echo "Error: save_dir is required"
    echo "Usage: ./pretrain_distributed.sh --save_dir /path/to/save [options]"
    exit 1
fi

# Validate data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist"
    exit 1
fi

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Calculate total world size
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# Construct restore file argument if provided
RESTORE_ARG=""
if [ ! -z "$RESTORE_FILE" ]; then
    RESTORE_ARG="checkpoint.restore_file=$RESTORE_FILE"
fi

# Run distributed training
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    fairseq_cli/hydra_train.py \
    --config-dir EAT/config \
    --config-name pretraining_AS2M \
    common.user_dir=$EAT_DIR \
    checkpoint.save_dir=$SAVE_DIR \
    $RESTORE_ARG \
    distributed_training.distributed_world_size=$WORLD_SIZE \
    optimization.max_update=$NUM_UPDATES \
    optimization.lr=[$LEARNING_RATE] \
    dataset.batch_size=$BATCH_SIZE \
    task.data=$DATA_DIR \
    task.h5_format=false
