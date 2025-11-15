import streamlit as st
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Legal RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

if not Path(CHROMA_PATH).exists():
    try:
        from src.ingestion.loader import load_and_chunk_all_documents
        from src.ingestion.vectorstore import build_vectorstore

        with st.spinner("🔨 Building vector database ... This may take a few minutes."):
            all_chunked_docs = load_and_chunk_all_documents()
            build_vectorstore(all_chunked_docs)

        st.success("✅ Vector database built successfully!")

    except Exception as e:
        st.error(f"❌ Error building vector database: {e}")
        st.stop()

from src.agents.orchestrator import run_query

st.title(" Criminal Law Research Assistant")
st.markdown("""
Ask questions about Indian Criminal Law including:
- **IPC** (Indian Penal Code)
- **CrPC** (Criminal Procedure Code)
- **Evidence Act**
- **Case Law & Judgments**
- **Regulations & Rules**
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(" Multi-agent system processing your question..."):
            try:
                answer = run_query(prompt)

                st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.sidebar:
    st.markdown("## :material/gavel: About")

    with st.container(border=True):
        st.markdown("### Multi-Agent RAG System")
        st.markdown("""
        This assistant uses a **sophisticated multi-agent architecture** to answer questions about Indian Criminal Law.
        """)

    with st.container(border=True):
        st.markdown("### :material/library_books: Knowledge Base")
        st.markdown("""
        - **Statutes** - IPC, CrPC, Evidence Act, POCSO, SC/ST Act
        - **Case Law** - Landmark Supreme Court judgments
        - **Regulations** - Prison Manual, Police Act
        """)

    with st.container(border=True):
        st.markdown("### :material/settings: Actions")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.caption("Powered by LangChain, Groq & ChromaDB")
