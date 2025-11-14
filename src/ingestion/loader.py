import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdfs_from_directory(directory_path: str, doc_type: str) -> List[Document]:
    documents = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata['doc_type'] = doc_type
                doc.metadata['source'] = filename
                documents.append(doc)

    return documents

def chunk_documents(documents: List[Document], doc_type: str) -> List[Document]:
    if doc_type == "statutes":
        chunk_size = 1000
        chunk_overlap = 200
    elif doc_type == "case_laws":
        chunk_size = 800
        chunk_overlap = 150
    else:
        chunk_size = 1000
        chunk_overlap = 200

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

def load_and_chunk_all_documents() -> Dict[str, List[Document]]:
    from src.config import DATA_PATHS

    all_chunked_docs = {}

    for doc_type, path in DATA_PATHS.items():
        print(f"Loading {doc_type} from {path}...")
        docs = load_pdfs_from_directory(path, doc_type)
        print(f"Loaded {len(docs)} pages from {doc_type}")

        print(f"Chunking {doc_type}...")
        chunks = chunk_documents(docs, doc_type)
        print(f"Created {len(chunks)} chunks from {doc_type}")

        all_chunked_docs[doc_type] = chunks

    return all_chunked_docs
