## Maybe Helpful
# https://huggingface.co/sentence-transformers

DEVICE = "cpu"

#### Storage Config
PERSIST_DIRECTORY = "./chroma_db"
CACHE_DIR = "./models"
DOCS_DIRECTORY = "./docs"

#### DB & Query Config (LLAMA2)
SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"
HF_ACCESS_TOKEN = ""

#### Output Config
USE_TOP_K_SIMILAR_DOC = 1

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know.

{context}

Question: {question}
Answer: 
""".strip()