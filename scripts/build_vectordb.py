import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import load_and_chunk_all_documents
from src.ingestion.vectorstore import build_vectorstore

print("Starting vector database build process...")

all_chunked_docs = load_and_chunk_all_documents()

build_vectorstore(all_chunked_docs)

print("Vector database build complete!")
