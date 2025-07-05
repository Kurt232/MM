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
