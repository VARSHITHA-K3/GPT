import os
from chromadb.config import Settings
from chromadb.api.fastapi import FastAPI
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
MODEL_ID = "TheBloke/orca_mini_3B-GGML"
MODEL_BASENAME = "orca-mini-3b.ggmlv3.q4_0.bin"
#MODEL_PATH = "C:/Users/dell/.cache/huggingface/hub/models--TheBloke--orca_mini_3B-GGML/snapshots/423525dd58bf01fca2a7ef57b7f8f39d4a9366b8/orca-mini-3b.ggmlv3.q4_0.bin"

# MODEL_TYPE="GPT4All"
# MODEL_PATH="models/orca-mini-3b.ggmlv3.q4_0.bin"