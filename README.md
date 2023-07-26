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

### Web-Application
A simple web application is made to make querying simpler.

To run it, make sure `flask` is installed by doing `pip install Flask`
```sh
python3 flask_main.py
```

Then, head to [http://127.0.0.1:5022/app/](http://127.0.0.1:5022/app/)

### Command line
Manually modifying source code MAY be needed. (in the `configs/` folder)

Steps to add and query documents:

1. `adddata.py` will automatically import `docs/*` files into Chroma db. And it is not recommended to put the same file into `docs/` directory because it will be reimported.
```sh
python3 adddata.py
```

2. `viewdata.py` to view loaded documents
```sh
python3 viewdata.py
```

3. `query.py` starts Querying the database!
```sh
python3 query.py

## Example
(cmd)> query
[*] Type in your query: What option did I think the employee should choose?

# `reload_qa` is a special command to reload the prompt specified in config.PROMPT_TEMPLATE
(cmd)> reload_qa
```

## Notes
Inspired by a project `PrivateGPT`.