import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")

DATA_PATHS = {
    "statutes": "data/statutes",
    "case_laws": "data/case_laws",
    "regulations": "data/regulations"
}

COLLECTION_NAMES = {
    "statutes": "statutes_collection",
    "case_laws": "cases_collection",
    "regulations": "regulations_collection"
}
