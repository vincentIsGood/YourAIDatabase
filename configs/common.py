# cpu, cuda, mps
DEVICE = "cpu"

### Storage Config
PERSIST_DIRECTORY = "./chroma_db"
CACHE_DIR = "./models"
DOCS_DIRECTORY = "./docs"

### Model Config
LOCAL_FILES_ONLY = False
HF_ACCESS_TOKEN = ""

### Query Config
USE_TOP_K_SIMILAR_DOC = 1