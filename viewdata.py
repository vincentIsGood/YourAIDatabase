import argparse
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import configs.common as config
import configs.llama2 as model_config
from utils.InteractiveConsole import InteractiveConsole, SimpleCommandHandler

cmdParser = argparse.ArgumentParser()
cmdParser.add_argument("-id", "--docid", default=None, type=str, help="Print content of a document")
cmdParser.add_argument("-it", "--interactive", action="store_true", default=False, help="Interactive mode")
cmdParsed = cmdParser.parse_args()

print("[+] Preparing Chroma DB")
embedding_func = SentenceTransformerEmbeddings(model_name=model_config.SENTENCE_EMBEDDING_MODEL, cache_folder=config.CACHE_DIR)
chromadb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embedding_func)

print("[+] Chroma index:")
print(chromadb._collection, "\n")

print("[+] Chroma # of collections: ", chromadb._collection.count())

def viewAllDocs():
    collection = chromadb._collection.get(include=["metadatas"])
    ids = collection["ids"]
    metadatas = collection["metadatas"]

    print("[+] Documents (ID -> Metadata)")
    for i in range(len(ids)):
        print(f"[*] '{ids[i]}': {metadatas[i]}")

def viewSpecificDoc(id: str):
    print("[+] Showing content for doc with id: %s" % id)
    collection = chromadb._collection.get(ids=[id], include=["metadatas", "documents"])
    print(collection["metadatas"])
    print(collection["documents"])

if cmdParsed.interactive:
    print("[+] Entering interactive mode")
    console = InteractiveConsole()
    console.addHandler(SimpleCommandHandler(lambda args: viewSpecificDoc(args[0]), "docid", "view document with content with its ID"))
    console.addHandler(SimpleCommandHandler(lambda args: viewAllDocs(), "docs", "view all documents with its content"))
    console.takeover()
elif cmdParsed.docid:
    viewSpecificDoc(cmdParsed.docid)
else: 
    viewAllDocs()