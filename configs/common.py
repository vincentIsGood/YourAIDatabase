from . import llama2_ggml as model_config
from .utils.module_utils import getCTransformersCudaLib_Windows

### Basic Config
## cpu, cuda, mps
# DEVICE = "cpu"
# CTRANSFORMERS_CUDA_LIB = None
DEVICE = "cuda"
CTRANSFORMERS_CUDA_LIB = getCTransformersCudaLib_Windows()

## GPU: LlamaCpp is a BIT faster and uses less memory.
## CPU: ctransformers has a huge edge on memory. (13b model RAM: << 1GB)
USE_LLAMACPP_INSTEAD_OF_CTRANSFORMERS = False

# use how much GPU (if enabled); `GPU_LAYERS = 20` (13b model: ~4GB. After query ~6GB)
# For Mac users using Metal.     `GPU_LAYERS = 1` is required
## ctransformers will get stuck waiting for more GPU, if value is too large
GPU_LAYERS = 20

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