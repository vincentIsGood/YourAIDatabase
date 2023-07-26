from . import llama2 as model_config

# cpu, cuda, mps
DEVICE = "cpu"
SKIP_PROMPT = True

### Storage Config
PERSIST_DIRECTORY = "./chroma_db"
CACHE_DIR = "./models"
DOCS_DIRECTORY = "./docs"

### Model Config
LOCAL_FILES_ONLY = True
HF_ACCESS_TOKEN = ""

### Query Config
USE_TOP_K_SIMILAR_DOC = 1