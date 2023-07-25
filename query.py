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

import configs.common as config
import configs.llama2 as model_config
from lib.utils.InteractiveConsole import InteractiveConsole, SimpleCommandHandler
from lib.AiDatabase import AiDatabaseQuerier

### Types
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import BaseLanguageModel
###

def main():
    aiDatabaseQuerier = AiDatabaseQuerier()

    def query(args):
        aiDatabaseQuerier.query(input("[*] Type in your query: "))

    def reload_qa(args):
        aiDatabaseQuerier.reloadPrompt()

    console = InteractiveConsole("(cmd)> ")
    console.addHandler(SimpleCommandHandler(query, "query", "Query your AI database"))
    console.addHandler(SimpleCommandHandler(reload_qa, "reload_qa", "Reload your prompt"))
    console.takeover()

main()