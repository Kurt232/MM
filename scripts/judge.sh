#!/bin/bash

NAME="0524_4k"

# Add cleanup function and trap to kill background processes on Ctrl+C
cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

START_TIME=$(date +%s)

tasks="helm|mmlu|0"
CUDA_VISIBLE_DEVICES="0" python llm-as-judge.py \
    "eval/$NAME" "./models/Llama3-8B" "latest" \
    "$tasks" > logs/$NAME/Llama3-8B.log 2>&1 &
CUDA_VISIBLE_DEVICES="1" python llm-as-judge.py "eval_judged" \
    "eval/$NAME/details/._models_Qwen2.5-14B/2025-05-25T19-37-04.422266" \
    "$tasks" > logs/$NAME/Qwen2.5-14B.log 2>&1 &

CUDA_VISIBLE_DEVICES="2" python llm-as-judge.py "eval_judged" \
    "eval/$NAME/details/._models_R1-Llama3-8B/2025-05-25T19-21-34.775637" \
    "$tasks" > logs/$NAME/R1-Llama3-8B.log 2>&1 &
CUDA_VISIBLE_DEVICES="3" python llm-as-judge.py "eval_judged" \
    "eval/$NAME/details/._models_R1-Qwen2.5-14B/2025-05-26T03-45-39.068645" \
    "$tasks" > logs/$NAME/R1-Qwen2.5-14B.log 2>&1 &

wait
END_TIME=$(date +%s)
TIME_COST=$((END_TIME - START_TIME))
echo "Time cost $((TIME_COST / 3600)):$(((TIME_COST % 3600) / 60)):$((TIME_COST % 60))"