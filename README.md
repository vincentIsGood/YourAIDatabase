# Your AI Database
Give your documents to the AI and let your LLM to analyze your documents on querying.

## How to use it
First, make sure you have `torch` installed. 

If you are using Mac, you may need to `export HNSWLIB_NO_NATIVE=1` before building chromadb.

Then, install project dependencies with 
```sh
pip install -r requirements.txt
```

Setup configuration in [`configs/common.py`](configs/common.py). Example:
```py
from . import llama2_ggml as model_config

LOCAL_FILES_ONLY = False

## Some models may need HuggingFace Token (eg. llama2)
HF_ACCESS_TOKEN = "hf_...."
```

If you want to use another model, create a new configuration (eg. [`configs/miniorca.py`](configs/miniorca.py)) and modify `configs/common.py` again.
```py
from . import miniorca as model_config
```

Manually modifying source code MAY be needed, if you want granular control of the application.

### Web-Application (Web UI)
A simple web application is made to make uploading and querying simpler.

To run it, make sure `flask` is installed by doing `pip install Flask`
```sh
python3 flask_main.py
```

Then, head to [http://127.0.0.1:5022/app/](http://127.0.0.1:5022/app/)

<details>
<summary>Notes on HTML-based resource</summary>

For *url upload*, it purely downloads online resources. If the website is not server-side rendered, you get nothing out of the url. (eg. Reddit)

If you really want to grab the content of a webpage, you are recommended to save the page as `pdf` on a browser.
</details>

### Command line
Steps to add and query documents:

1. `adddata.py` will automatically import `docs/*` files into Chroma db. And it is not recommended to put the same file into `docs/` directory because it will be reimported.
```sh
python3 adddata.py
```

2. `viewdata.py` to view loaded documents
```sh
python3 viewdata.py
```

3. `query.py` to query the database!
```sh
python3 query.py

## Example
(cmd)> query
[*] Type in your query: What option did I think the employee should choose?

# `reload_qa` is a special command to reload the prompt specified in config.PROMPT_TEMPLATE
(cmd)> reload_qa
```

## GPU Support
You have 2 options: `llama-cpp-python` or `ctransformers`. The difference is as follows:
```py
## GPU: LlamaCpp is a BIT faster and uses less memory.
## CPU: ctransformers has a huge edge on memory. (13b model RAM: << 1GB)
```

<details>
<summary>Mac Metal Problem</summary>

When you encounter compilation problems while loading library `ggml-metal.metal`, like `Error: Use of undeclared identifier 'assert'`, `constant int64_t`, blah blah blah.

Try to upgrade your python. Currently, I am using `python 3.10.12` (upgrade with pyenv or whatever)
</details>

### `llama-cpp-python`
According to [LangChain](https://python.langchain.com/docs/integrations/llms/llamacpp)'s instruction:
```sh
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

For Mac (Apple Silicon):
```sh
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

Then modify `configs/common.py`
```py
DEVICE = "cpu"
USE_LLAMACPP_INSTEAD_OF_CTRANSFORMERS = True
```

### `ctransformers` with `CUDA`
Follow [official](https://github.com/marella/ctransformers#gpu) instructions:
```sh
CT_CUBLAS=1 pip install ctransformers --upgrade --force-reinstall --no-binary ctransformers
```

For MAC:
```sh
CT_METAL=1 pip install ctransformers --upgrade --force-reinstall --no-binary ctransformers
```

<details>
<summary>Can't work? Try this.</summary>

You may have to set the path to `ctransformers.dll` or `ctransformers.so` if ctransformers still don't work.

If `pip install ctransformers` itself doesn't enable GPU support, you can directly download `.dll` or `.so` from creator's [GitHub](https://github.com/marella/ctransformers/tree/main/ctransformers/lib/cuda). And put the file into folder `.../Python39/Lib/site-packages/ctransformers/lib/cuda/` (create `lib/cuda` directory if needed)

Then modify `configs/common.py`:
```py
CTRANSFORMERS_CUDA_LIB = r"...\Python39\Lib\site-packages\ctransformers\lib\cuda\ctransformers"
```

Or use the following 2 provided functions.

For Windows (modify `configs/common.py`):
```py
from .utils.module_utils import getCTransformersCudaLib_Windows

DEVICE = "cuda"
CTRANSFORMERS_CUDA_LIB = getCTransformersCudaLib_Windows()
```

For unix (modify `configs/common.py`):
```py
from .utils.module_utils import getCTransformersCudaLib_Unix

DEVICE = "cuda"
CTRANSFORMERS_CUDA_LIB = getCTransformersCudaLib_Unix()
```
</details>

## Caution
The web application provided in this project is meant to work on localhost. 
Do not try to expose it to the public.

## Notes
Used: Chromadb, LangChain, Llama Cpp Python, Ctransformers, HuggingFace, Flask

Inspired by a project called `PrivateGPT`.