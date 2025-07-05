#!/bin/bash

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

root="data/test"

for name in $root/*; do
    if [ -d "$name" ]; then
        name=$(basename "$name")
        echo "Processing $name"
        mkdir -p $root/$name/logs
        {
            python src/metric_instruct.py $root/$name ./models/R-Phi4 latest >> $root/$name/logs/R-Phi4.log
            python src/metric_instruct.py $root/$name ./models/R1-Llama3-8B latest >> $root/$name/logs/R1-Llama3-8B.log
            python src/metric_instruct.py $root/$name ./models/I-Phi4 latest > $root/$name/logs/I-Phi4.log
        } &
    fi
done

wait
echo "All evaluations completed"
