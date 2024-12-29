#!/bin/bash

# Default values
CUDA_DEVICE="0"
BATCH_SIZE=96
MODEL_PATH=""
SAVE_DIR=""
DATA_PATH=""
TARGET_LENGTH=1024
MIXUP=0.8
MASK_RATIO=0.2
PREDICTION_MODE="CLS_TOKEN"
RESTORE_FILE=""
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/labeled"
LABELS_FILE="${DATA_DIR}/labels.csv"
WEIGHTS_FILE=""
EAT_DIR="${PROJECT_DIR}/EAT"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
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
        --target_length)
            TARGET_LENGTH="$2"
            shift 2
            ;;
        --mixup)
            MIXUP="$2"
            shift 2
            ;;
        --mask_ratio)
            MASK_RATIO="$2"
            shift 2
            ;;
        --prediction_mode)
            PREDICTION_MODE="$2"
            shift 2
            ;;
        --restore_file)
            RESTORE_FILE="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --labels)
            LABELS_FILE="$2"
            shift 2
            ;;
        --weights)
            WEIGHTS_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$MODEL_PATH" ] || [ -z "$SAVE_DIR" ]; then
    echo "Error: model_path and save_dir are required parameters"
    echo "Usage: ./finetune.sh --model_path /path/to/model --save_dir /path/to/save [options]"
    exit 1
fi

# Validate data directory and labels file
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist"
    exit 1
fi

if [ ! -f "$LABELS_FILE" ]; then
    echo "Error: Labels file $LABELS_FILE does not exist"
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Construct restore file argument if provided
RESTORE_ARG=""
if [ ! -z "$RESTORE_FILE" ]; then
    RESTORE_ARG="checkpoint.restore_file=$RESTORE_FILE"
fi

# Run training
python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning_anuraset \
    checkpoint.save_dir=$SAVE_DIR \
    checkpoint.best_checkpoint_metric=mAP \
    common.user_dir=$EAT_DIR \
    $RESTORE_ARG \
    dataset.batch_size=$BATCH_SIZE \
    task.data=$DATA_DIR \
    task.h5_format=true \
    task.AS2M_finetune=true \
    task.target_length=$TARGET_LENGTH \
    task.roll_aug=true \
    task.weights_file=$WEIGHTS_FILE \
    model.model_path=$MODEL_PATH \
    model.mixup=$MIXUP \
    ++model.mask_ratio=$MASK_RATIO \
    model.prediction_mode=PredictionMode.$PREDICTION_MODE
