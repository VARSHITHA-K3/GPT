import os
from chromadb.config import Settings
#load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/db"
SOURCE_DIRECTORY = "C:/Users/dell/Documents/Downloads"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl = 'duckdb+parquet',
        persist_directory = PERSIST_DIRECTORY,
        anonymized_telemetry = False
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_ID = "nomic-ai/gpt4all-falcon-ggml"
MODEL_BASENAME = "ggml-model-gpt4all-falcon-q4_0.bin"


MODEL_TYPE="GPT"
MODEL_PATH="C:/Users/dell/Documents/GPT models/orca-mini-3b.ggmlv3.q4_0.bin"
MODEL_N_CTX=1000
MODEL_N_BATCH=8
TARGET_SOURCE_CHUNKS=4