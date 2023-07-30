import os
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

from .loaders import TextTrimmedLoader
from .utils import file_utils

## https://github.com/Unstructured-IO/unstructured#document-parsing
### Types
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
###

DEFAULT_LOADER: 'Tuple[Type[BaseLoader], object]' = (TextLoader, {"encoding": None})
SUPPORTED_LOADERS: 'dict[str, Tuple[Type[BaseLoader], object]]' = {
    "txt": (TextTrimmedLoader, {"encoding": None}),
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

class LocalFileLoader:
    def __init__(self, encoding = "utf-8", useDefaultLoader = False):
        """
            useDefaultLoader: use default loader if an unknown file extension is encountered
        """
        self.encoding = encoding
        self.useDefaultLoader = useDefaultLoader
        self.loadedDocs: 'List[Document]' = []
        self.textSplitter = CharacterTextSplitter(
            chunk_size=4000, 
            chunk_overlap=0)

    def loadDoc(self, filePath, source = None):
        """source: If not None, overwrite the metadata "source" of docs with this value
        """
        ext = file_utils.fileExt(filePath)
        if ext in SUPPORTED_LOADERS:
            loaderClassType, loaderArgs = SUPPORTED_LOADERS[file_utils.fileExt(filePath)]
        else: 
            if not self.useDefaultLoader:
                print("[!] Cannot find loader for file '%s'. Ignoring it." % filePath)
                return
            print("[!] Cannot find loader for file '%s'. Using default loader." % filePath)
            loaderClassType, loaderArgs = DEFAULT_LOADER
        
        if "encoding" in loaderArgs:
            loaderArgs["encoding"] = self.encoding
        
        # loader = TextLoader(pathToTxt, encoding=self.encoding)
        loader = loaderClassType(filePath, **loaderArgs)
        docs = loader.load()

        if source:
            for doc in docs:
                doc.metadata = {"source": source}

        # print(doc)
        for doc in self.textSplitter.split_documents(docs):
            self.loadedDocs.append(doc)

    def getDocs(self):
        return self.loadedDocs


import shutil
import mimetypes
import requests
from .utils.randutils import randomString

class WebFileLoader(LocalFileLoader):
    """
    from lib.DocLoader import WebFileLoader
    loader = WebFileLoader()
    loader.loadWebDoc("https://docs.python.org/3/library/mimetypes.html")
    """
    def __init__(self, tmpFolder = "./tmp"):
        super().__init__("utf-8")
        self.tmpDir = tmpFolder
        if not os.path.exists(tmpFolder):
            os.mkdir(tmpFolder, mode=755)
        
    def loadWebDoc(self, url):
        print("[+] Trying to download doc: ", url)
        try:
            res = requests.get(url)
            baseFilename = os.path.basename(res.url)
            if baseFilename == "":
                baseFilename = randomString()
            if not "." in baseFilename:
                baseFilename + mimetypes.guess_extension(res.headers.get("content-type"))
            
            outFilename = self.tmpDir + "/" + baseFilename
            print("[+] Saving doc to ", outFilename)
            with open(outFilename, "wb+") as f:
                f.write(res.content)
            
            self.loadDoc(outFilename, res.url)
        except Exception:
            print("[-] Cannot add document")

    def cleanupTmp(self):
        print("[+] Removing tmp directory: ", self.tmpDir)
        shutil.rmtree(self.tmpDir)