import chromadb
import os
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

def retrieve_statutes(query: str, k: int = 5) -> List[Document]:
    """
    Retrieves relevant sections from bare acts/statutes.

    Args:
        query: User's legal question
        k: Number of top results to return

    Returns:
        List of Document objects with statute chunks and metadata
    """
    collection = chroma_client.get_collection(name="statutes_collection")

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    documents = []
    for i in range(len(results['documents'][0])):
        doc = Document(
            page_content=results['documents'][0][i],
            metadata=results['metadatas'][0][i] if results['metadatas'] else {}
        )
        documents.append(doc)

    return documents


def retrieve_cases(query: str, k: int = 5) -> List[Document]:
    """
    Retrieves relevant case law excerpts.

    Args:
        query: User's legal question
        k: Number of top results to return

    Returns:
        List of Document objects with case law chunks and metadata
    """
    collection = chroma_client.get_collection(name="cases_collection")

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    documents = []
    for i in range(len(results['documents'][0])):
        doc = Document(
            page_content=results['documents'][0][i],
            metadata=results['metadatas'][0][i] if results['metadatas'] else {}
        )
        documents.append(doc)

    return documents


def retrieve_regulations(query: str, k: int = 5) -> List[Document]:
    """
    Retrieves relevant government regulations.

    Args:
        query: User's legal question
        k: Number of top results to return

    Returns:
        List of Document objects with regulation chunks and metadata
    """
    collection = chroma_client.get_collection(name="regulations_collection")

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    documents = []
    for i in range(len(results['documents'][0])):
        doc = Document(
            page_content=results['documents'][0][i],
            metadata=results['metadatas'][0][i] if results['metadatas'] else {}
        )
        documents.append(doc)

    return documents
