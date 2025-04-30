# model
BASE_PATH=${1:-"."} # 当前目录修改为相对路径
CKPT_NAME="gpt2-base"
CKPT="${BASE_PATH}/student_models/${CKPT_NAME}/"
# CKPT="gpt2-xl" # download automatically


# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
TEACHER_NAME="gpt2"
TEACHER_CKPT="${BASE_PATH}/teacher_models/${TEACHER_NAME}/"
# length
MAX_LENGTH=512

#TRAIN
CRITIC_TIME=3
LAMBDA_GP=10
DO_TRAIN=1      # Set to 1 to enable training
DO_EVAL=1       # Set to 1 to enable evaluation
BATCH_SIZE=48 # Define Batch Size (Increased from 32, keeping Grad Checkpoint)
NUM_WORKERS=8  # Define Num Workers
GRADIENT_CHECKPOINTING=1 # Set to 1 to enable gradient checkpointing
GENERATE_PLOT=1 # Set to 1 to generate t-SNE plot, 0 to disable

# Hyperparameters (explicitly defined here)
LEARNING_RATE="5e-6"  # Learning rate (previously calculated internally)
GRAD_ACCUM_STEPS=1    # Gradient accumulation steps (previously hardcoded/default)
WEIGHT_DECAY="1e-2"   # Weight decay (previously default)
WARMUP_ITERS=0        # Warmup iterations (previously default)
# EPOCHS=3            # Full run epochs (now controlled below for testing)

#INFERENCE
MAX_INPUT_LEN=768
DO_SAMPLE=0     # Set to 1 to enable sampling during generation


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_NAME}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --dev-num 1000"
# TRAIN
OPTS+=" --critic-time ${CRITIC_TIME}"
OPTS+=" --lambda-gp ${LAMBDA_GP}"
OPTS+=" --eval-gen 1" # Note: This seems hardcoded to 1, corresponding argument might be missing or named differently
# HP (add here)
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --num-workers ${NUM_WORKERS}"
OPTS+=" --lr ${LEARNING_RATE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACCUM_STEPS}"
OPTS+=" --weight-decay ${WEIGHT_DECAY}"
OPTS+=" --warmup-iters ${WARMUP_ITERS}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"

# Boolean flags (conditionally added)
if [ ${DO_TRAIN} -eq 1 ]; then
  OPTS+=" --do-train"
fi
if [ ${DO_EVAL} -eq 1 ]; then
  OPTS+=" --do-eval"
fi
if [ ${GRADIENT_CHECKPOINTING} -eq 1 ]; then
  OPTS+=" --gradient-checkpointing"
fi
if [ ${DO_SAMPLE} -eq 1 ]; then
  OPTS+=" --do-sample"
fi

# Conditionally add the plot generation flag
if [ ${GENERATE_PLOT} -eq 1 ]; then
  OPTS+=" --generate-tsne-plot"
fi

# Add DeepSpeed arguments (Commented out to revert to FP32/DDP)
# OPTS+=" --deepspeed"
# OPTS+=" --deepspeed_config ds_config_bf16.json"

# Settings for a quick test run (comment out the next two lines for a full run)
OPTS+=" --train-num 10000" # Use only 10000 training samples
OPTS+=" --epochs 1"        # Train for only 1 epoch

#inference
OPTS+=" --max-input-len ${MAX_INPUT_LEN}"
# Note: --do-sample is handled above conditionally


#!/bin/bash


# 设置分布式训练参数
# DISTRIBUTED_ARGS="--nproc_per_node=5 --master_port=29500" # 单机多卡
DISTRIBUTED_ARGS="--nproc_per_node=1 --master_port=29500" # 单机单卡

# 设置 PYTHONPATH 环境变量
export PYTHONPATH=${BASE_PATH}

# 打印将要执行的命令（用于调试）
echo "Running command: torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/try_sh.py ${OPTS} $@"

# Define log file path with timestamp (relative to script location after cd)
LOG_DIR="output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

# Create output directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Execute the training script and pipe stdout/stderr to tee
# tee writes to the log file AND displays on the terminal
{
  torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/try_sh.py ${OPTS} "$@" 2>&1 # Redirect stderr to stdout *within* the block for tee
} | tee "${LOG_FILE}"

