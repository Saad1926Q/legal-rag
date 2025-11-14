import chromadb
from typing import List, Dict
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_PATH, COLLECTION_NAMES, EMBEDDING_MODEL

def create_vectorstore():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    collections = {}
    for doc_type, collection_name in COLLECTION_NAMES.items():
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        collections[doc_type] = collection

    return client, collections, embedding_model

def add_documents_to_collection(collection, documents: List[Document], embedding_model):
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [f"{doc.metadata.get('source', 'unknown')}_{i}" for i, doc in enumerate(documents)]

    embeddings = embedding_model.encode(texts).tolist()

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

def build_vectorstore(all_chunked_docs: Dict[str, List[Document]]):
    print("Creating ChromaDB collections...")
    client, collections, embedding_model = create_vectorstore()

    for doc_type, documents in all_chunked_docs.items():
        print(f"Adding {len(documents)} chunks to {doc_type} collection...")
        add_documents_to_collection(collections[doc_type], documents, embedding_model)
        print(f"Completed {doc_type} collection")

    print("Vector database built successfully!")
    return client
