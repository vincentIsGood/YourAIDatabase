## Maybe Helpful
# https://huggingface.co/sentence-transformers

SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "psmathur/orca_mini_v2_7b"

PROMPT_TEMPLATE = """
### System:
You are an AI assistant that helps people find information.

### User:
Does the given information contain any answers to the following question?

If you don't know the answer, just say that you don't know.

{question}

Justification is not needed.

### Input:
{context}

### Response:
""".strip()