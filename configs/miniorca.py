## Maybe Helpful
# https://huggingface.co/sentence-transformers

SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "psmathur/orca_mini_v2_7b"
IS_GGML = False

IS_LLM_LOCAL = False
IS_TOKENIZER_LOCAL = False

PROMPT_TEMPLATE = """
### System:
You are an AI assistant that helps people find information. 
If you don't know the answer, just say that you don't know. 
Justification is not needed.

### User:
Does the given information contains any answers to the following question?

{question}

### Input:
{context}

### Response:
""".strip()