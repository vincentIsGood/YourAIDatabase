import os
import sys
import shutil

import configs.common as config
import configs.llama2 as model_config
from lib.docloader import Loaders

def filenameNoExt(nonDirFile: str):
    dotIndex = nonDirFile.rfind(".")
    if dotIndex != -1:
        return nonDirFile[:dotIndex]
    return nonDirFile

def loadData():
    loader = Loaders("utf-8")

    importedDir = config.DOCS_DIRECTORY + "/imported"
    if not os.path.isdir(importedDir):
        os.mkdir(importedDir, mode=755)

    for file in os.listdir(config.DOCS_DIRECTORY):
        if filenameNoExt(file).endswith("_ignore"):
            continue
        docFilename = config.DOCS_DIRECTORY + "/" + file
        if os.path.isdir(docFilename):
            continue
        print("[+] Loading data into chroma: ", docFilename)
        loader.loadDoc(docFilename)
        shutil.move(docFilename, importedDir)

    return loader.getDocs()


loadedDocs = loadData()
if len(loadedDocs) == 0:
    print("[+] No files to be imported")
    sys.exit(0)

print("[+] Preparing Chroma DB")
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embedding_func = SentenceTransformerEmbeddings(model_name=model_config.SENTENCE_EMBEDDING_MODEL, cache_folder=config.CACHE_DIR)
chromadb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embedding_func)
chromadb.add_documents(documents=loadedDocs)