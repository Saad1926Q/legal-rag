import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Dict, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from tools.retrieval_tools import retrieve_statutes, retrieve_cases, retrieve_regulations
from agents.citation_agent import extract_citations
from agents.response_agent import generate_response
from langchain_core.tools import tool

load_dotenv()


class AgentState(TypedDict):
    """State passed between agents in the graph"""
    question: str
    messages: List
    retrieved_docs: Dict[str, List[Document]]
    citations: str
    final_answer: str


# Define retrieval tools
@tool
def search_statutes(query: str) -> str:
    """Searches through bare acts and statutes related to Indian criminal law. Use this when the question is about laws, sections, or legal provisions from acts like IPC (Indian Penal Code), CrPC (Criminal Procedure Code), Evidence Act, etc."""
    docs = retrieve_statutes(query, k=5)
    return f"Retrieved {len(docs)} statute documents"

@tool
def search_cases(query: str) -> str:
    """Searches through criminal court judgments and case law. Use this when the question asks about precedents, judicial interpretations, or specific court rulings in criminal matters."""
    docs = retrieve_cases(query, k=5)
    return f"Retrieved {len(docs)} case law documents"

@tool
def search_regulations(query: str) -> str:
    """Searches through government regulations and rules related to criminal law. Use this for questions about regulatory compliance, administrative rules, or government notifications in the criminal law domain."""
    docs = retrieve_regulations(query, k=5)
    return f"Retrieved {len(docs)} regulation documents"

tools = [search_statutes, search_cases, search_regulations]

llm = ChatGroq(
    model=os.getenv("LLM_MODEL", "llama-3.1-70b-versatile"),
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

llm_with_tools = llm.bind_tools(tools)


def retrieval_agent_node(state: AgentState) -> AgentState:
    """
    NODE 1: Retrieval agent decides whether to use tools or respond directly.

    If question is about criminal law → calls retrieval tools
    If question is greeting/irrelevant → responds directly without tools

    Reads from state: question
    Writes to state: messages, retrieved_docs
    """
    question = state["question"]

    system_message = """You are a legal research assistant specializing EXCLUSIVELY in Indian Criminal Law.

Your expertise covers:
- Indian Penal Code (IPC)
- Criminal Procedure Code (CrPC)
- Indian Evidence Act
- Criminal case law and judgments
- Criminal law regulations

Your task:
1. If the question is a GREETING (hello, hi, how are you, etc.):
   - Respond politely and briefly WITHOUT using any tools
   - Introduce yourself as a criminal law specialist

2. If the question is about topics OUTSIDE Indian Criminal Law (civil law, divorce, contracts, general knowledge, etc.):
   - Politely decline WITHOUT using any tools
   - Explain your specialization

3. If the question IS about Indian Criminal Law:
   - Use the retrieval tools to fetch relevant documents
   - You can call MULTIPLE tools if needed (statutes, cases, regulations)
   - After calling tools, acknowledge what you retrieved

Available tools:
- search_statutes: For IPC, CrPC, Evidence Act sections
- search_cases: For court judgments and precedents
- search_regulations: For government rules and regulations"""

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=question)
    ]

    response = llm_with_tools.invoke(messages)

    messages.append(response)
    state["messages"] = messages

    if response.tool_calls:
        retrieved_docs = {
            "statutes": [],
            "cases": [],
            "regulations": []
        }

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "search_statutes":
                retrieved_docs["statutes"] = retrieve_statutes(tool_args["query"], k=5)
            elif tool_name == "search_cases":
                retrieved_docs["cases"] = retrieve_cases(tool_args["query"], k=5)
            elif tool_name == "search_regulations":
                retrieved_docs["regulations"] = retrieve_regulations(tool_args["query"], k=5)

        state["retrieved_docs"] = retrieved_docs
    else:
        state["final_answer"] = response.content

    return state


def citation_node(state: AgentState) -> AgentState:
    """
    NODE 2: Extract citations using the citation agent.

    Reads from state: question, retrieved_docs
    Writes to state: citations
    """
    question = state["question"]
    retrieved_docs = state["retrieved_docs"]

    citations = extract_citations(question, retrieved_docs)

    state["citations"] = citations

    return state


def response_node(state: AgentState) -> AgentState:
    """
    NODE 3: Generate final response using the response agent.

    Reads from state: question, retrieved_docs, citations
    Writes to state: final_answer
    """
    question = state["question"]
    retrieved_docs = state["retrieved_docs"]
    citations = state["citations"]

    final_answer = generate_response(question, retrieved_docs, citations)

    state["final_answer"] = final_answer

    return state


def should_continue_to_citations(state: AgentState) -> str:
    """
    CONDITIONAL EDGE: Check if tools were called.

    If tools were called → retrieved_docs will have content → go to citations
    If no tools called → final_answer already set → end
    """
    retrieved_docs = state.get("retrieved_docs", {})

    # Check if any documents were retrieved
    has_docs = any(len(docs) > 0 for docs in retrieved_docs.values())

    if has_docs:
        return "citations"
    else:
        return "end"


workflow = StateGraph(AgentState)

workflow.add_node("retrieval_agent", retrieval_agent_node)
workflow.add_node("citations", citation_node)
workflow.add_node("response", response_node)

workflow.set_entry_point("retrieval_agent")

workflow.add_conditional_edges(
    "retrieval_agent",
    should_continue_to_citations,
    {
        "citations": "citations",
        "end": END
    }
)

workflow.add_edge("citations", "response")
workflow.add_edge("response", END)

app = workflow.compile()


def run_query(question: str) -> str:
    """
    Main function to run the multi-agent orchestrator.

    Args:
        question: User's question

    Returns:
        Final answer string
    """
    initial_state = {
        "question": question,
        "messages": [],
        "retrieved_docs": {},
        "citations": "",
        "final_answer": ""
    }

    final_state = app.invoke(initial_state)

    return final_state["final_answer"]
