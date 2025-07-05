#!/bin/bash

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

NAME="0623_32k"
MAX_LEN=32768
# Create logs directory if it doesn't exist
mkdir -p data/eval/$NAME
mkdir -p data/eval/$NAME/logs
{
    MODEL=R-Phi4
    CUDA_VISIBLE_DEVICES=2 python src/infer_instruct.py data/eval/$NAME ./models/$MODEL --max_length $MAX_LEN --tensor_parallel_size 4 --timestamp latest > data/eval/$NAME/logs/$MODEL.log
    python src/metric_instruct.py data/eval/$NAME ./models/$MODEL latest >> data/eval/$NAME/logs/$MODEL.log
    python src/eval.py data/eval/$NAME ./models/$MODEL latest >> data/eval/$NAME/logs/$MODEL.log # add the metric results in the generation logs
}

{
    MODEL=R1-Llama3-8B
    CUDA_VISIBLE_DEVICES=3 python src/infer_instruct.py data/eval/$NAME ./models/$MODEL --max_length $MAX_LEN --tensor_parallel_size 4 --timestamp latest > data/eval/$NAME/logs/$MODEL.log
    python src/metric_instruct.py data/eval/$NAME ./models/$MODEL latest >> data/eval/$NAME/logs/$MODEL.log
    python src/eval.py data/eval/$NAME ./models/$MODEL latest >> data/eval/$NAME/logs/$MODEL.log # add the metric results in the generation logs
}

# Wait for all background jobs to complete
wait
echo "All evaluations completed"
python src/show_results.py "data/eval/$NAME"
