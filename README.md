# Your AI Database
Give your documents to the AI and let your LLM to analyze your documents on querying.

## How to use it
Command line is not extensively suppoorted. Manually modifying source code MAY be needed. (at the time of writing this document)

Steps to add and query documents:

1. `adddata.py` will automatically import `docs/*` files into Chroma db. And it is not recommended to put the same file into `docs/` directory because it will be reimported.
```py
python3 adddata.py
```

2. `viewdata.py` to view loaded documents
```py
python3 viewdata.py
```

3. `query.py` starts Querying the database!
```py
python3 query.py

## Example
(Query)> What option did I think the employee should choose?

# `reload_qa` is a special command to reload the prompt in config.PROMPT_TEMPLATE
(Query)> reload_qa
```