import importlib
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

import configs.llama2 as config

### Types
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import BaseLanguageModel
###

def createLLM():
    if len(config.HF_ACCESS_TOKEN) > 0:
        huggingface_hub.login(token=config.HF_ACCESS_TOKEN)
        langmodel = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, cache_dir=config.CACHE_DIR, use_auth_token=True)
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL, cache_dir=config.CACHE_DIR, use_auth_token=True)
    else:
        langmodel = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, cache_dir=config.CACHE_DIR)
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL, cache_dir=config.CACHE_DIR)

    # https://huggingface.co/docs/transformers/main/main_classes/pipelines
    # Streaming Output
    # https://github.com/hwchase17/langchain/issues/2918
    # https://github.com/hwchase17/langchain/issues/4950
    pipeline = transformers.pipeline(
        "text-generation",
        model=langmodel,
        tokenizer=tokenizer,
        streamer=TextStreamer(tokenizer),

        device_map=config.DEVICE,
        max_length=1000,
        do_sample=True,
        top_k=10,
        eos_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={
            "temperature": 0,
        },
        callbacks=[StreamingStdOutCallbackHandler()]
    )

def createRetrievalQA(llm: BaseLanguageModel, retriever: VectorStoreRetriever):
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs={
            "prompt": PromptTemplate(template=config.PROMPT_TEMPLATE, input_variables=["context", "question"])
        },
        return_source_documents=True
    )

def main():
    embedding_func = SentenceTransformerEmbeddings(model_name=config.SENTENCE_EMBEDDING_MODEL, cache_folder=config.CACHE_DIR)
    chromadb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embedding_func)
    print("[+] Chroma # of collections: ", chromadb._collection.count())

    retriever = chromadb.as_retriever(search_kwargs={"k": config.USE_TOP_K_SIMILAR_DOC, "include_metadata": True})
    llm = createLLM()
    retrievalQA = createRetrievalQA(llm, retriever)

    while True:
        userInput = input("(Query)> ").strip()
        if userInput == "reload_qa":
            importlib.reload(config)
            retrievalQA = createRetrievalQA(llm, retriever)
            continue
        
        res = retrievalQA({"query": userInput})
        # print(res["result"])
        for source in res["source_documents"]:
            print(source.metadata)

main()