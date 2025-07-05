#!/bin/bash

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM
gpus="2,3"
root="data/test/0-1k"
root1="data/merge_bench" # target saved path

mkdir -p $root1/logs
for model in ./merged1/*; do # eval all in merged
    if [ -d "$model" ]; then
        model_name=$(basename "$model")
        echo "Processing model: $model_name"
        if [ -f $root/logs/$model_name.log ]; then
            echo "already eval $model_name, skipping..."
            continue
        fi
        {
            CUDA_VISIBLE_DEVICES=$gpus python src/infer_load.py $root ./models/R-Phi4 $model latest --save_dir $root1 --max_length 2048 --tensor_parallel_size 2 > $root1/logs/$model_name.log
            python src/metric_instruct.py $root1 $model latest >> $root1/logs/$model_name.log &
            python src/eval.py $root1 $model latest &
        }
    fi
done

wait

echo "All evaluations completed"
