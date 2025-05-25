#!/bin/bash

GPUS="2"
export CUDA_VISIBLE_DEVICES=$GPUS
# export VLLM_WORKER_MULTIPROC_METHOD="spawn"

MODEL_ARGS="configs/vllm_r1_llama8b.yaml"
OUTPUT_DIR=eval/0523_4k_100/$MODEL

# lighteval|simpleqa|0|0,\
tasks="\
helm|mmlu|0|0,\
lighteval|drop|0|0,\
extended|ifeval|0|0,\
helm|truthfulqa|0|0,\
helm|bold:race|0|0,\
helm|commonsenseqa|0|0,\
helm|piqa|0|0,\
helm|openbookqa|0|0,\
lighteval|arc:easy|0|0,\
lighteval|gpqa:diamond|0|0,\
leaderboard|arc:challenge|0|0,\
lighteval|aime24|0|0,\
lighteval|aime25|0|0,\
lighteval|math_500|0|0,\
lighteval|gsm8k|0|0,\
extended|lcb:codegeneration_release_v6|0|0"

START_TIME=$(date +%s)
# max_sample: 1000
lighteval vllm $MODEL_ARGS "$tasks" \
    --max-samples 100 \
    --save-details \
    --dataset-loading-processes 32 \
    --output-dir $OUTPUT_DIR

END_TIME=$(date +%s)
TIME_COST=$((END_TIME - START_TIME))
echo "Time cost $((TIME_COST / 3600)):$(((TIME_COST % 3600) / 60)):$((TIME_COST % 60))"