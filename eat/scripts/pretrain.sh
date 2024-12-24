#!/bin/bash

# Default values
CUDA_DEVICES="0"
BATCH_SIZE=12
SAVE_DIR=""
DATA_PATH=""
WORLD_SIZE=1
RESTORE_FILE=""
TARGET_LENGTH=1024
MASK_RATIO=0.75
NUM_UPDATES=400000
LEARNING_RATE=1.5e-4
UPDATE_FREQ=1
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/unlabeled"

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
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --world_size)
            WORLD_SIZE="$2"
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
        --data_dir)
            DATA_DIR="$2"
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
    echo "Usage: ./pretrain.sh --save_dir /path/to/save [options]"
    exit 1
fi

# Validate data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist"
    exit 1
fi

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Construct restore file argument if provided
RESTORE_ARG=""
if [ ! -z "$RESTORE_FILE" ]; then
    RESTORE_ARG="checkpoint.restore_file=$RESTORE_FILE"
fi

# Run pretraining
python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name pretraining_AS2M \
    common.user_dir=EAT \
    checkpoint.save_dir=$SAVE_DIR \
    $RESTORE_ARG \
    distributed_training.distributed_world_size=$WORLD_SIZE \
    optimization.max_update=$NUM_UPDATES \
    optimization.lr[0]=$LEARNING_RATE \
    optimization.update_freq[0]=$UPDATE_FREQ \
    dataset.batch_size=$BATCH_SIZE \
    task.data=$DATA_DIR \
    task.h5_format=true \
    task.target_length=$TARGET_LENGTH \
    model.mask_ratio=$MASK_RATIO