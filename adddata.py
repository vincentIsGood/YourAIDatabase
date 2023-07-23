import os
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import configs.llama2 as config
from lib.docloader import Loaders

def loadData():
    loader = Loaders()
    for file in os.listdir(config.DOCS_DIRECTORY):
        docFilename = config.DOCS_DIRECTORY + "/" + file
        print("[+] Loading data into chroma: ", docFilename)
        loader.loadDoc(docFilename)
    return loader.getDocs()

embedding_func = SentenceTransformerEmbeddings(model_name=config.SENTENCE_EMBEDDING_MODEL, cache_folder=config.CACHE_DIR)
chromadb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embedding_func)
chromadb.add_documents(documents=loadData())