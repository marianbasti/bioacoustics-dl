#!/bin/bash

# Default values
CUDA_DEVICE="0"
BATCH_SIZE=96
TARGET_LENGTH=1024
MIXUP=0.8
MASK_RATIO=0.2
PREDICTION_MODE="CLS_TOKEN"
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/labeled"
EAT_DIR="${PROJECT_DIR}/EAT"

# Required parameters with no defaults
MODEL_PATH=""
SAVE_DIR=""
METADATA_DIR=""
WEIGHTS_FILE=""
RESTORE_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda|--batch_size|--model_path|--save_dir|--target_length|--mixup|--mask_ratio|--prediction_mode|--restore_file|--data_dir|--weights|--metadata_dir)
            varname=$(echo ${1#--} | tr '-' '_')
            eval ${varname}="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
for param in MODEL_PATH SAVE_DIR; do
    if [ -z "${!param}" ]; then
        echo "Error: ${param,,} is required"
        exit 1
    fi
done

# Validate data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist"
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Construct optional arguments
OPTIONAL_ARGS=""
[[ -n "$RESTORE_FILE" ]] && OPTIONAL_ARGS="$OPTIONAL_ARGS checkpoint.restore_file=$RESTORE_FILE"
[[ -n "$METADATA_DIR" ]] && OPTIONAL_ARGS="$OPTIONAL_ARGS task.metadata_dir=$METADATA_DIR"
[[ -n "$WEIGHTS_FILE" ]] && OPTIONAL_ARGS="$OPTIONAL_ARGS task.weights_file=$WEIGHTS_FILE"

# Run training
python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning_anuraset \
    checkpoint.save_dir=$SAVE_DIR \
    checkpoint.best_checkpoint_metric=mAP \
    common.user_dir=$EAT_DIR \
    dataset.batch_size=$BATCH_SIZE \
    task.data=$DATA_DIR \
    task.h5_format=true \
    task.AS2M_finetune=true \
    task.target_length=$TARGET_LENGTH \
    task.roll_aug=true \
    model.model_path=$MODEL_PATH \
    model.mixup=$MIXUP \
    ++model.mask_ratio=$MASK_RATIO \
    model.prediction_mode=PredictionMode.$PREDICTION_MODE \
    $OPTIONAL_ARGS
