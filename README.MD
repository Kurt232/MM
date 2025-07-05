# Setup
```sh
uv venv --python 3.10 vv
source vv/bin/activate && uv pip install --upgrade pip
deactivate

git clone git@github.com:Kurt232/MM.git
cd MM
git submodule update --init

# cuda 12.2
uv pip install vllm --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
uv pip install -e lighteval[vllm,math] --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
uv pip install "datasets==2.20.0" --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
uv pip install ipykernel ipywidgets --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
# Description
## overview
We use inference and evaluation separate architecture.
`src/infer_base.py` and `src/infer_instruct.py` to generate responses.
`src/metric_base.py` and `src/metric_instruct.py` to eval the responses.
`src/eval.py` to add the metric value in the response logs.

## benchmark
We prepared 9 benchmark for *completion* prompt and *instruct* prompt.
```python
# completion prompt
"""\
mm|mmlu_pro_c|0|0,\
mm|truthfulqa_c|0|0,\
mm|commonsenseqa_c|0|0,\
mm|arc_easy_c|0|0,\
mm|arc_challenge_c|0|0,\
mm|gpqa_diamond_c|0|0,\
mm|aime24_c|0|0,\
mm|math_500_c|0|0,\
mm|gsm8k_c|0|0\
"""
# instruct prompt
"""\
mm|mmlu_pro|0|0,\
mm|truthfulqa|0|0,\
mm|commonsenseqa|0|0,\
mm|arc_easy|0|0,\
mm|arc_challenge|0|0,\
mm|gpqa_diamond|0|0,\
mm|aime24|0|0,\
mm|math_500|0|0,\
mm|gsm8k|0|0\
"""
```

## inference
`infer_base.py` and `infer_instruct.py`, where the args of sampling_generation and sample_num are hard code in the script. 

## evaluation
`src/metric_base.py` and `src/metric_instruct.py` to eval the responses.
`src/eval.py` to add the metric value in the response logs.

# Recipes
See it in the `scripts/infer.sh`.
We recommend to use soft link to save models in `./models/`

```bash
python src/infer_instruct.py data/eval/$NAME ./models/$MODEL --max_length $MAX_LEN --tensor_parallel_size 4 --timestamp latest > data/eval/$NAME/logs/$MODEL.log
python src/metric_instruct.py data/eval/$NAME ./models/$MODEL latest >> data/eval/$NAME/logs/$MODEL.log
python src/eval.py data/eval/$NAME ./models/$MODEL latest >> data/eval/$NAME/logs/$MODEL.log # add the metric results in the generation logs
```

# Run the benchmark
We generate the data in `data/eval/0623_32k` from instruct prompt datasets including `mm|truthfulqa|0|0, mm|commonsenseqa|0|0, mm|arc_easy|0|0, mm|arc_challenge|0|0, mm|gpqa_diamond|0|0, mm|aime24|0|0, mm|math_500|0|0, mm|gsm8k|0|0`. Note that we exclude `mm|mmlu_pro|0|0`.

we can see the details in `scripts/merge_bench.sh` and `scripts/eval_bench.sh`
```bash
git clone https://huggingface.co/datasets/Kurt232/MM data # or git clone https://hf-mirror.com/datasets/Kurt232/MM

output_dir="data/test/0-1k"
model="./merged1/llama_linear_1" # can be replaced any merged model

python src/infer_load.py \
$output_dir \ # generations dir
./models/R-Phi4 \ # load which model\'s generations
$model \ # current inference model
latest \ # timestamp for generations
[--save_dir dir] # we can change where to save, and the default is $output_dir
--max_length 2048 --tensor_parallel_size 2 > $root/logs/$model_name.log

python src/metric_instruct.py $root $model latest >> $root/logs/$model_name.log
python src/eval.py $root $model latest
```

# Prepare data
The data has already been prepared by `git clone https://huggingface.co/datasets/Kurt232/MM data`

```bash
bash scripts/bench.sh # to generate 32k context length
# ! please masked the mm|mmlu_pro|0|0 in src/infer_instruct.py
python split.py # split reasoning data
python split_map.py # map base model data according to reasoning data.
bash scripts/split.sh # to get the metrics
```
