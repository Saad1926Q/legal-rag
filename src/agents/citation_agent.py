import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model=os.getenv("LLM_MODEL", "llama-3.1-70b-versatile"),
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

system_message = """You are a legal citation expert. Your task is to analyze retrieved legal documents and extract relevant citations that should be referenced when answering the user's question.

For each relevant document, create a properly formatted citation including:
- Document name (from source metadata)
- Page number (from page metadata)
- Brief reason why it's relevant

Only cite documents that are actually relevant to answering the question. Not all retrieved documents need to be cited.

Format your response as:

**STATUTE CITATIONS:**
1. [Document name], Page [X] - [Brief relevance reason]
2. ...

**CASE LAW CITATIONS:**
1. [Case name], Page [X] - [Brief relevance reason]
2. ...

**REGULATION CITATIONS:**
1. [Regulation name], Page [X] - [Brief relevance reason]
2. ...

If a category has no relevant citations, write "None"."""


def extract_citations(question: str, retrieved_docs: Dict[str, List[Document]]) -> str:
    """
    Extract and format relevant citations from retrieved documents.

    Args:
        question: The user's original question
        retrieved_docs: Dict with keys 'statutes', 'cases', 'regulations' mapping to lists of Documents

    Returns:
        Formatted citation string from LLM
    """

    # Build the prompt with all retrieved documents
    prompt = f"Question: {question}\n\n"
    prompt += "Retrieved Legal Documents:\n\n"

    statute_docs = retrieved_docs.get("statutes", [])
    case_docs = retrieved_docs.get("cases", [])
    regulation_docs = retrieved_docs.get("regulations", [])

    if statute_docs:
        prompt += "=== STATUTES ===\n"
        for i, doc in enumerate(statute_docs):
            source = doc.metadata.get("source", "Unknown").replace(".pdf", "")
            page = doc.metadata.get("page", "")
            page_str = f"Page {int(page) + 1}" if page != "" else "Page N/A"

            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content

            prompt += f"\n[Statute {i+1}] {source}, {page_str}\n{content_preview}\n"

    if case_docs:
        prompt += "\n=== CASE LAW ===\n"
        for i, doc in enumerate(case_docs):
            source = doc.metadata.get("source", "Unknown").replace(".pdf", "")
            page = doc.metadata.get("page", "")
            page_str = f"Page {int(page) + 1}" if page != "" else "Page N/A"

            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content

            prompt += f"\n[Case {i+1}] {source}, {page_str}\n{content_preview}\n"

    if regulation_docs:
        prompt += "\n=== REGULATIONS ===\n"
        for i, doc in enumerate(regulation_docs):
            source = doc.metadata.get("source", "Unknown").replace(".pdf", "")
            page = doc.metadata.get("page", "")
            page_str = f"Page {int(page) + 1}" if page != "" else "Page N/A"

            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content

            prompt += f"\n[Regulation {i+1}] {source}, {page_str}\n{content_preview}\n"

    prompt += "\n\nAnalyze these documents and provide relevant citations for answering the question."

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    return response.content
