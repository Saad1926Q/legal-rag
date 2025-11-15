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

system_message = """You are a legal research assistant specializing in Indian Criminal Law. Your task is to provide accurate, well-structured answers to legal questions.

Guidelines:
- Use the provided legal documents as your source of truth
- Reference specific sections, case names, and regulations when applicable
- Structure your answer clearly with proper legal reasoning
- Be precise and cite sources appropriately
- If the documents don't fully answer the question, acknowledge the limitations
- Write in a professional but accessible tone

The citations have already been extracted and provided to you. Use them to support your answer."""


def generate_response(question: str, retrieved_docs: Dict[str, List[Document]], citations: str) -> str:
    """
    Generate final answer using retrieved documents and extracted citations.

    Args:
        question: The user's original question
        retrieved_docs: Dict with keys 'statutes', 'cases', 'regulations' mapping to lists of Documents
        citations: Formatted citations from citation agent

    Returns:
        Final answer as string
    """

    context = ""

    statute_docs = retrieved_docs.get("statutes", [])
    case_docs = retrieved_docs.get("cases", [])
    regulation_docs = retrieved_docs.get("regulations", [])

    if statute_docs:
        context += "\n=== STATUTE EXCERPTS ===\n"
        for doc in statute_docs:
            source = doc.metadata.get("source", "Unknown").replace(".pdf", "")
            page = doc.metadata.get("page", "")
            context += f"\n[{source}, Page {int(page) + 1 if page != '' else 'N/A'}]\n"
            context += f"{doc.page_content}\n"
            context += "-" * 80 + "\n"

    if case_docs:
        context += "\n=== CASE LAW EXCERPTS ===\n"
        for doc in case_docs:
            source = doc.metadata.get("source", "Unknown").replace(".pdf", "")
            page = doc.metadata.get("page", "")
            context += f"\n[{source}, Page {int(page) + 1 if page != '' else 'N/A'}]\n"
            context += f"{doc.page_content}\n"
            context += "-" * 80 + "\n"

    if regulation_docs:
        context += "\n=== REGULATION EXCERPTS ===\n"
        for doc in regulation_docs:
            source = doc.metadata.get("source", "Unknown").replace(".pdf", "")
            page = doc.metadata.get("page", "")
            context += f"\n[{source}, Page {int(page) + 1 if page != '' else 'N/A'}]\n"
            context += f"{doc.page_content}\n"
            context += "-" * 80 + "\n"

    prompt = f"""Question: {question}

Legal Document Context:
{context}

Relevant Citations (already extracted):
{citations}

Based on the above legal documents and citations, provide a comprehensive answer to the question. Structure your response clearly and reference the sources appropriately."""

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    final_response = response.content

    if citations and citations.strip():
        final_response += f"\n\n{citations}"

    return final_response
