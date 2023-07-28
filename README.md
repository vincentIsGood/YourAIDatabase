# Your AI Database
Give your documents to the AI and let your LLM to analyze your documents on querying.

## How to use it
First, make sure you have `torch` installed. 

Then, install project dependencies with 
```sh
pip install -r requirements.txt
```

Setup configuration in [`configs/common.py`](configs/common.py):
```py
# By default, llama2 is used. Hence the following is required and cannot be empty
from . import llama2 as model_config

LOCAL_FILES_ONLY = False
HF_ACCESS_TOKEN = "hf_...."
```

If you want to use another model, create a new configuration (eg. [`configs/miniorca.py`](configs/miniorca.py)) and modify `configs/common.py` again.
```py
from . import miniorca as model_config
```

Manually modifying source code MAY be needed, if you want granular control of the application.

### Web-Application
A simple web application is made to make uploading and querying simpler.

To run it, make sure `flask` is installed by doing `pip install Flask`
```sh
python3 flask_main.py
```

Then, head to [http://127.0.0.1:5022/app/](http://127.0.0.1:5022/app/)

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

## Workaround with `ctransformers` using `CUDA`
I had to specify the following in the program in order to use `CUDA` with `ctransformers`.
```py
# Exclude the extension of the library (eg. dll, so)
# Example:
LLM(
    ...
    lib=r"...\Python39\Lib\site-packages\ctransformers\lib\cuda\ctransformers",
    ...
)
```

To simplify the process (modify `configs/common.py`):
```py
# Download instructions: https://github.com/marella/ctransformers#cuda
# If `pip install` doesn't work, try to simply copy `.../lib/cuda/ctransformers.dll` into the corresponding location
CTRANSFORMERS_CUDA_LIB = r"...\Python39\Lib\site-packages\ctransformers\lib\cuda\ctransformers"
```

## Notes
Used: Chromadb, LangChain, HuggingFace, Flask

Inspired by a project called `PrivateGPT`.