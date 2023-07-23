from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import configs.llama2 as config

embedding_func = SentenceTransformerEmbeddings(model_name=config.SENTENCE_EMBEDDING_MODEL, cache_folder=config.CACHE_DIR)
chromadb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embedding_func)
print("[+] Chroma # of collections: ", chromadb._collection.count())
print("[+] Chroma collections:")
print(chromadb._collection, "\n")

collection = chromadb._collection.get(include=["metadatas"])
ids = collection["ids"]
metadatas = collection["metadatas"]

print("[+] Documents (ID -> Metadata)")
for i in range(len(ids)):
    print(f"[*] '{ids[i]}': {metadatas[i]}")