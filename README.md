# Your AI Database
Give your documents to the AI and let your LLM to analyze your documents on querying.

## How to use it
Command line is not extensively suppoorted. Manually modifying source code MAY be needed. (at the time of writing this document)

`query.py`
`viewdata.py`

Steps to add and query documents:

1. `adddata.py` will automatically import `docs/*` files into Chroma db, remember to delete / remove the document from `docs/` directory after importing the documents (to prevent them from being loaded again)
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
(Query)> I want to know when was the meeting about aws conducted

# `reload_qa` is a special command to reload the prompt in config.PROMPT_TEMPLATE
(Query)> reload_qa
```