#!/bin/bash

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

NAME="0527_4k"
# Create logs directory if it doesn't exist
mkdir -p data/eval/$NAME
mkdir -p data/eval/$NAME/logs


{
set -e
CUDA_VISIBLE_DEVICES="0" python src/infer.py data/eval/$NAME ./models/Qwen2.5-14B > data/eval/$NAME/logs/Qwen2.5-14B.log 2>&1
python src/metric.py data/eval/$NAME ./models/Qwen2.5-14B latest >> data/eval/$NAME/logs/Qwen2.5-14B.log
} &

{
set -e
CUDA_VISIBLE_DEVICES="1" python src/infer.py data/eval/$NAME ./models/R1-Qwen2.5-14B > data/eval/$NAME/logs/R1-Qwen2.5-14B.log 2>&1
python src/metric.py data/eval/$NAME ./models/R1-Qwen2.5-14B latest >> data/eval/$NAME/logs/R1-Qwen2.5-14B.log
} &

{
set -e
CUDA_VISIBLE_DEVICES="2" python src/infer.py data/eval/$NAME ./models/Llama3-8B > data/eval/$NAME/logs/Llama3-8B.log 2>&1
python src/metric.py data/eval/$NAME ./models/Llama3-8B latest >> data/eval/$NAME/logs/Llama3-8B.log
} &

{
set -e
CUDA_VISIBLE_DEVICES="3" python src/infer.py data/eval/$NAME ./models/R1-Llama3-8B > data/eval/$NAME/logs/R1-Llama3-8B.log 2>&1
python src/metric.py data/eval/$NAME ./models/R1-Llama3-8B latest >> data/eval/$NAME/logs/R1-Llama3-8B.log
} &

# Wait for all background jobs to complete
wait
echo "All evaluations completed"