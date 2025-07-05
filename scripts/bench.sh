#!/bin/bash

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

NAME="0623_32k"
MAX_LEN=$((1024 * 32)) # actually 8k
# Create logs directory if it doesn't exist
mkdir -p data/eval/$NAME
mkdir -p data/eval/$NAME/logs
{
    CUDA_VISIBLE_DEVICES=2 python src/infer_instruct.py data/eval/$NAME ./models/R-Phi4 --max_length $MAX_LEN --tensor_parallel_size 1 --timestamp latest > data/eval/$NAME/logs/R-Phi4.log
    python src/metric_instruct.py data/eval/$NAME ./models/R-Phi4 latest >> data/eval/$NAME/logs/R-Phi4.log
} &

{
    CUDA_VISIBLE_DEVICES=3 python src/infer_instruct.py data/eval/$NAME ./models/R1-Llama3-8B --max_length $MAX_LEN --tensor_parallel_size 1 --timestamp latest > data/eval/$NAME/logs/R1-Llama3-8B.log
    python src/metric_instruct.py data/eval/$NAME ./models/R1-Llama3-8B latest >> data/eval/$NAME/logs/R1-Llama3-8B.log
} &

{
    CUDA_VISIBLE_DEVICES="0,1" python src/infer_instruct.py data/eval/$NAME ./models/I-Phi4 --max_length $MAX_LEN --tensor_parallel_size 2 --timestamp latest > data/eval/$NAME/logs/I-Phi4.log
    python src/metric_instruct.py data/eval/$NAME ./models/I-Phi4 latest >> data/eval/$NAME/logs/I-Phi4.log
} &

# Wait for all background jobs to complete
wait
echo "All evaluations completed"
python src/show_results.py "data/eval/$NAME"
