from typing import List
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

### Types
from langchain.schema import Document
###

class Loaders:
    def __init__(self):
        self.loadedDocs: 'List[Document]' = []
        self.textSplitter = CharacterTextSplitter(
            chunk_size=4000, 
            chunk_overlap=0
        )

    def loadDoc(self, pathToTxt):
        self.loadedDocs.append(*self.textSplitter.split_documents(TextLoader(pathToTxt).load()))

    def getDocs(self):
        return self.loadedDocs