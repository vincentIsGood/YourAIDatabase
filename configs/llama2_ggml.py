## Maybe Helpful
# https://huggingface.co/sentence-transformers

SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "TheBloke/Llama-2-7B-Chat-GGML"
IS_GGML = True

## Manually download (LLM_MODEL_TYPE can be found in 'config.json')
# LLM_MODEL_TYPE = "llama"
# LLM_MODEL = "/path/to/ggml-gpt-2.bin"
## OR
# LLM_MODEL = "/path/to/ggml-gpt-2/folder"  # contains 'config.json' & 'model.bin'

IS_LLM_LOCAL = False

# https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5
PROMPT_TEMPLATE = """
<s>
[INST]
<<SYS>>
If you don't know the answer, just say that you don't know. You must provide a concise answer.
<</SYS>>

Use the following pieces of information to answer the question.
[/INST] 
Information: 
{context}
</s>

<s> [INST] {question} [/INST] 
""".strip() + "\n\n"