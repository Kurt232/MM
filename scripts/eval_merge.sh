#!/bin/bash

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

load_dir="data/test/0-1k"
save_dir="data/1111" # target saved path
model="./merge/linearmerge_0.5"
mkdir -p $save_dir/logs

model_name=$(basename "$model")

CUDA_VISIBLE_DEVICES="0" python src/infer_load.py \
    $load_dir ./models/R-Phi4 $model latest \
    --save_dir $save_dir \
    --max_length 2048 \
    --tensor_parallel_size 1 > $save_dir/logs/$model_name.log

python src/metric_instruct.py $save_dir $model latest >> $save_dir/logs/$model_name.log &
python src/eval.py $save_dir $model latest &

wait

