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
EAT_DIR="${PROJECT_DIR}/EAT"

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

# Count number of GPUs from CUDA_DEVICES
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_DEVICES"
WORLD_SIZE=${#GPU_ARRAY[@]}

# Construct restore file argument if provided
RESTORE_ARG=""
if [ ! -z "$RESTORE_FILE" ]; then
    RESTORE_ARG="checkpoint.restore_file=$RESTORE_FILE"
fi

# Remove the python command at the end and just export the variables
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export WORLD_SIZE=${#GPU_ARRAY[@]}
export RESTORE_ARG
if [ ! -z "$RESTORE_FILE" ]; then
    RESTORE_ARG="checkpoint.restore_file=$RESTORE_FILE"
fi
