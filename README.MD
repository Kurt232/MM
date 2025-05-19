# env

```sh
uv venv --python 3.10 vv
source vv/bin/activate && uv pip install --upgrade pip
deactivate

git clone git@github.com:huggingface/lighteval.git
# cuda 12.2
uv pip install vllm --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
uv pip install -e lighteval[vllm,math] --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
uv pip install "datasets==2.20.0" --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
uv pip install ipykernel --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```
