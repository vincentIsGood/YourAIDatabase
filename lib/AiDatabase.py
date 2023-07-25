import importlib
import sys
import torch
import huggingface_hub
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

import configs.common as config
import configs.llama2 as model_config
from .output_callbacks import StreamingCallbackHandler

### Types
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import BaseLanguageModel
###

class AiDatabaseQuerier:
    def __init__(self):
        print("[+] Preparing Chroma DB")
        embedding_func = SentenceTransformerEmbeddings(model_name=model_config.SENTENCE_EMBEDDING_MODEL, cache_folder=config.CACHE_DIR)
        self.chromadb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embedding_func)
        print("[+] Chroma # of collections: ", self.chromadb._collection.count())

        self.retriever = self.chromadb.as_retriever(search_kwargs={"k": config.USE_TOP_K_SIMILAR_DOC, "include_metadata": True})
        self.llm = createLLM()
        self.retrievalQA = createRetrievalQA(self.llm, self.retriever)

    def reloadPrompt(self):
        importlib.reload(model_config)
        self.retrievalQA = createRetrievalQA(self.llm, self.retriever)

    def query(self, queryStr: str):
        if queryStr == "": 
            return

        res = self.retrievalQA({"query": queryStr})
        return res["source_documents"]
        # print(res["result"])
        # for source in res["source_documents"]:
        #     print(source.metadata)

def createLLM():
    print("[+] Loading LLM model")
    if len(config.HF_ACCESS_TOKEN) > 0:
        huggingface_hub.login(token=config.HF_ACCESS_TOKEN)
        langmodel = AutoModelForCausalLM.from_pretrained(
            model_config.LLM_MODEL, cache_dir=config.CACHE_DIR, local_files_only=config.LOCAL_FILES_ONLY, use_auth_token=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.LLM_MODEL, cache_dir=config.CACHE_DIR, local_files_only=config.LOCAL_FILES_ONLY, use_auth_token=True)
    else:
        langmodel = AutoModelForCausalLM.from_pretrained(
            model_config.LLM_MODEL, cache_dir=config.CACHE_DIR, local_files_only=config.LOCAL_FILES_ONLY)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.LLM_MODEL, cache_dir=config.CACHE_DIR, local_files_only=config.LOCAL_FILES_ONLY)

    # https://huggingface.co/docs/transformers/main/main_classes/pipelines
    # Streaming Output
    # https://github.com/hwchase17/langchain/issues/2918
    # https://github.com/hwchase17/langchain/issues/4950
    pipeline = transformers.pipeline(
        "text-generation",
        model=langmodel,
        tokenizer=tokenizer,
        streamer=TextStreamer(tokenizer, skip_prompt=config.SKIP_PROMPT),

        device_map=config.DEVICE,
        max_length=1000,
        do_sample=True,
        top_k=10,
        eos_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={
            "temperature": 0.5,
        },
        callbacks=[StreamingCallbackHandler(sys.stdout)]
    )

def createRetrievalQA(llm: BaseLanguageModel, retriever: VectorStoreRetriever):
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs={
            "prompt": PromptTemplate(template=model_config.PROMPT_TEMPLATE, input_variables=["context", "question"])
        },
        return_source_documents=True
    )
