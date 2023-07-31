from . import llama2_ggml as model_config
from .utils.module_utils import getCTransformersCudaLib_Windows

### Basic Config
## cpu, cuda, mps
DEVICE = "cpu"
CTRANSFORMERS_CUDA_LIB = None
# DEVICE = "cuda"
# CTRANSFORMERS_CUDA_LIB = getCTransformersCudaLib_Windows()

# use how much GPU (if enabled)
GPU_LAYERS = 50

SKIP_PROMPT = True

### WebApp Config
WEB_UPLOAD_SECRET = "asldjhukdhuiddygkjdrhg"

### Storage Config
PERSIST_DIRECTORY = "./chroma_db"
CACHE_DIR = "./models"
DOCS_DIRECTORY = "./docs"

### Model Config
HF_ACCESS_TOKEN = ""

### Query Config
USE_TOP_K_SIMILAR_DOC = 1