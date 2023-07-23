## Maybe Helpful
# https://huggingface.co/sentence-transformers

SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know.

{context}

Question: {question}
Answer: 
""".strip()