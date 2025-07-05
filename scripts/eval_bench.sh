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
mkdir -p $root/logs
for model in ./merged/*; do # eval all in merged
    if [ -d "$model" ]; then
        model_name=$(basename "$model")
        echo "Processing model: $model_name"
        if [ -f $root/logs/$model_name.log ]; then
            echo "already eval $model_name, skipping..."
            continue
        fi
        {
            CUDA_VISIBLE_DEVICES=$gpus python src/infer_load.py $root ./models/R-Phi4 $model latest --max_length 2048 --tensor_parallel_size 2 > $root/logs/$model_name.log
            python src/metric_instruct.py $root $model latest >> $root/logs/$model_name.log &
            python src/eval.py $root $model latest &
        }
    fi
done

wait

python src/show_results.py "$root" > "$root/logs/show_results1.log"
echo "All evaluations completed"
