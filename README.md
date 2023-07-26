# Your AI Database
Give your documents to the AI and let your LLM to analyze your documents on querying.

## How to use it
First, make sure you have `torch` installed. 

Then, install project dependencies with 
```sh
pip install -r requirements.txt
```

### Web-Application
A simple web application is made to make querying simpler.

To run it, make sure `flask` is installed with something like `pip install Flask`
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
What option did I think the employee should choose?

# `reload_qa` is a special command to reload the prompt specified in config.PROMPT_TEMPLATE
(cmd)> reload_qa
```

## Notes
Inspired by a project `PrivateGPT`.