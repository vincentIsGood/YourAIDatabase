## Maybe Helpful
# https://huggingface.co/sentence-transformers

SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "TheBloke/Llama-2-13B-chat-GGML"
LLM_MODEL_BIN_FILE = "llama-2-13b-chat.ggmlv3.q3_K_S.bin"
IS_GGML = True

## Manually download (LLM_MODEL_TYPE can be found in 'config.json')
# LLM_MODEL_TYPE = "llama"
# LLM_MODEL = "/path/to/ggml-gpt-2.bin"
## OR
# LLM_MODEL = "/path/to/ggml-gpt-2/folder"  # contains 'config.json' & 'model.bin'

IS_LLM_LOCAL = False

PROMPT_TEMPLATE = """
### SYSTEM: 
Please behave and help the user. 
If you don't know the answer, just say that you don't know. 
Otherwise, you must provide a concise answer.

### USER: 
Use the following pieces of info to answer the last question:

Info starts here:

{context}

Info ends here.

Last question: {question}

### ASSISTANT: 
""".strip() + "\n"