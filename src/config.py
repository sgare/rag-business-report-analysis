# File Paths
PDF_PATH = "data/apple_hbr.pdf"
DB_PERSIST_PATH = "db_chroma"

# Text Splitter Config
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 64

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM Model Config
MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# LlamaCPP Config (Hardware settings)
# Change n_gpu_layers to 0 if you want to run on CPU
N_GPU_LAYERS = 40      # Number of layers to offload to GPU 
N_BATCH = 512          # Batch size for prompt processing 
N_CTX = 4096           # Context window size 
MAX_TOKENS = 1024      # Max tokens to generate