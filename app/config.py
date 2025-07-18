import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from a .env file (if exists)

PINECONE_API_KEY = os.getenv(                 #  Pinecone API Key
    "PINECONE_API_KEY",
    "Enter Pincecone API Keys"
)

INDEX_NAME = "simple-free-rag"  #Pinecone Index Configuration
INDEX_DIMENSION = 384

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # embedding & LLM Models
LLM_MODEL_NAME = "google/flan-t5-base"

