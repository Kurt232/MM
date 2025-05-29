#!/bin/bash
set -e

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

NAME=0530_2k
mkdir -p data/eval/$NAME
mkdir -p data/eval/$NAME/logs

START_TIME=$(date +%s)
NUM_GPUS=4
MODELS=($(ls "./merged"))
# Randomize the models array
MODELS=($(printf "%s\n" "${MODELS[@]}" | sort -R))
TOTAL_MODELS=${#MODELS[@]}

MODELS_PER_GPU=$(( (TOTAL_MODELS + NUM_GPUS - 1) / NUM_GPUS ))

process_inference() {
    local gpu=$1
    local start_idx=$2
    local end_idx=$3
    
    for ((i=start_idx; i<end_idx; i++)); do
        local MODEL=${MODELS[$i]}
        
        echo "Processing $MODEL on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python src/infer.py data/eval/$NAME ./merged/$MODEL > data/eval/$NAME/logs/$MODEL.log
        python src/metric.py data/eval/$NAME ./merged/$MODEL latest >> data/eval/$NAME/logs/$MODEL.log &
    done
}

# Process configs in parallel across GPUs
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    start_idx=$((gpu * MODELS_PER_GPU))
    end_idx=$((start_idx + MODELS_PER_GPU))
    
    # Ensure we don't go past the total number of configs
    if [ $start_idx -lt $TOTAL_MODELS ]; then
        if [ $end_idx -gt $TOTAL_MODELS ]; then
            end_idx=$TOTAL_MODELS
        fi
        process_inference $gpu $start_idx $end_idx &
    fi
done

# Wait for all GPU processes to finish
wait

COST_TIME=$(( $(date +%s) - $START_TIME ))
echo "All models processed in $(($COST_TIME/3600)):$(($COST_TIME%3600/60)):$(($COST_TIME%60))"