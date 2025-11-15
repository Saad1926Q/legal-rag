import streamlit as st
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.orchestrator import run_query

st.set_page_config(
    page_title="Criminal Law RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Criminal Law Research Assistant")
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
    st.header("About")
    st.info("""
    This assistant uses RAG (Retrieval Augmented Generation) to answer questions about Indian Criminal Law.

    It searches through:
    - Statutes (IPC, CrPC, Evidence Act)
    - Case Law & Judgments
    - Government Regulations

    """)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
