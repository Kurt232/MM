#!/bin/bash

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

NAME="b0/0602_4k"
# Create logs directory if it doesn't exist
mkdir -p data/eval/$NAME
mkdir -p data/eval/$NAME/logs


{
set -e
CUDA_VISIBLE_DEVICES="0" python src/infer_base.py data/eval/$NAME ./models/Llama3-8B --max_length 4096 > data/eval/$NAME/logs/Llama3-8B.log 2>&1
python src/metric_base.py data/eval/$NAME ./models/Llama3-8B latest >> data/eval/$NAME/logs/Llama3-8B.log
} &

{
set -e
CUDA_VISIBLE_DEVICES="1" python src/infer_base.py data/eval/$NAME ./models/Qwen2.5-14B --max_length 4096 > data/eval/$NAME/logs/Qwen2.5-14B.log 2>&1
python src/metric_base.py data/eval/$NAME ./models/Qwen2.5-14B latest >> data/eval/$NAME/logs/Qwen2.5-14B.log
} &

{
set -e
CUDA_VISIBLE_DEVICES="2" python src/infer_instruct.py data/eval/$NAME ./models/R1-Llama3-8B --max_length 4096 > data/eval/$NAME/logs/R1-Llama3-8B.log 2>&1
python src/metric_instruct.py data/eval/$NAME ./models/R1-Llama3-8B latest >> data/eval/$NAME/logs/R1-Llama3-8B.log
} &

{
set -e
CUDA_VISIBLE_DEVICES="3" python src/infer_instruct.py data/eval/$NAME ./models/R1-Qwen2.5-14B --max_length 4096 > data/eval/$NAME/logs/R1-Qwen2.5-14B.log 2>&1
python src/metric_instruct.py data/eval/$NAME ./models/R1-Qwen2.5-14B latest >> data/eval/$NAME/logs/R1-Qwen2.5-14B.log
} &

# {
# set -e
# CUDA_VISIBLE_DEVICES="0" python src/infer_instruct.py data/eval/$NAME ./models/I-Llama3-8B --max_length 4096 > data/eval/$NAME/logs/I-Llama3-8B.log 2>&1
# python src/metric_instruct.py data/eval/$NAME ./models/I-Llama3-8B latest >> data/eval/$NAME/logs/I-Llama3-8B.log
# } &

# {
# set -e
# CUDA_VISIBLE_DEVICES="1" python src/infer_instruct.py data/eval/$NAME ./models/I-Qwen2.5-14B --max_length 4096 > data/eval/$NAME/logs/I-Qwen2.5-14B.log 2>&1
# python src/metric_instruct.py data/eval/$NAME ./models/I-Qwen2.5-14B latest >> data/eval/$NAME/logs/I-Qwen2.5-14B.log
# } &

# Wait for all background jobs to complete
wait
echo "All evaluations completed"
python src/show_results.py "data/eval/$NAME"