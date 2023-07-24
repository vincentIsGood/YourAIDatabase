from typing import List, Type, Tuple
from langchain.document_loaders import (
    TextLoader, 
    CSVLoader, 
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import CharacterTextSplitter

import utils.FileUtils as FileUtils

## https://github.com/Unstructured-IO/unstructured#document-parsing
### Types
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
###

DEFAULT_LOADER: 'Tuple[Type[BaseLoader], object]' = (TextLoader, {"encoding": None})
SUPPORTED_LOADERS: 'dict[str, Tuple[Type[BaseLoader], object]]' = {
    "txt": (TextLoader, {"encoding": None}),
    "csv": (CSVLoader, {"encoding": None}),

    "xlsx": (UnstructuredExcelLoader, {}),
    "xls": (UnstructuredExcelLoader, {}),
    "docx": (UnstructuredWordDocumentLoader, {}),
    "doc": (UnstructuredWordDocumentLoader, {}),
    "pptx": (UnstructuredPowerPointLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),

    "pdf": (UnstructuredPDFLoader, {}),
    "md": (UnstructuredMarkdownLoader, {}),
    "xml": (UnstructuredXMLLoader, {}),
    "html": (UnstructuredHTMLLoader, {}),
}

class Loaders:
    def __init__(self, encoding = None):
        self.encoding = encoding
        self.loadedDocs: 'List[Document]' = []
        self.textSplitter = CharacterTextSplitter(
            chunk_size=4000, 
            chunk_overlap=0
        )

    def loadDoc(self, filePath):
        ext = FileUtils.fileExt(filePath)
        if ext in SUPPORTED_LOADERS:
            loaderClassType, loaderArgs = SUPPORTED_LOADERS[FileUtils.fileExt(filePath)]
        else: 
            print("[!] Cannot find loader for file '%s'. Using default loader." % filePath)
            loaderClassType, loaderArgs = DEFAULT_LOADER
        
        # TextLoader(pathToTxt, encoding=self.encoding)
        if "encoding" in loaderArgs:
            loaderArgs["encoding"] = self.encoding
        
        loader = loaderClassType(filePath, **loaderArgs)
        doc = loader.load()

        print(doc)
        self.loadedDocs.append(*self.textSplitter.split_documents(doc))

    def getDocs(self):
        return self.loadedDocs