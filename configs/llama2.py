## Maybe Helpful
# https://huggingface.co/sentence-transformers

SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"
IS_GGML = False

IS_LLM_LOCAL = False
IS_TOKENIZER_LOCAL = False

# https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5
PROMPT_TEMPLATE = """
<s>
[INST]
<<SYS>>
Please behave and help the user. 
If you don't know the answer, just say that you don't know. 
Otherwise, you must provide a concise answer.
<</SYS>>

Use the following pieces of information to answer the question.
[/INST] 
Information: 
{context}
</s>

<s> [INST] {question} [/INST] 
""".strip() + "\n\n"