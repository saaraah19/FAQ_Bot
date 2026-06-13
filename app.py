# app.py — orchestrator. Imports everything, calls everything. Nothing imports this.
import os

import streamlit as st
from dotenv import load_dotenv

from ingest import ingest
from retriever import build_retriever, retrieve
from generator import generate_stream

load_dotenv()

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="FAQ Bot", page_icon="💬", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f4f6f9; }

    .bot-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .bot-header h1 { margin: 0; font-size: 1.6rem; font-weight: 600; }
    .bot-header p  { margin: 0.2rem 0 0; font-size: 0.9rem; opacity: 0.85; }

    .doc-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1a73e8;
        border: 1px solid #c5e0f5;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }

    [data-testid="stChatMessage"] {
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ──────────────────────────────────────────────
def init_state():
    defaults = {
        "history":       [],
        "retriever":     None,
        "document_name": None,
        "show_upload":   True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    init_state()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="bot-header">
        <h1>💬 FAQ Bot</h1>
        <p>Upload a document and ask me anything about it.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Document badge — shows which file is loaded ───────────────────────────
    if st.session_state["document_name"]:
        st.markdown(
            f'<div class="doc-badge">📄 {st.session_state["document_name"]}</div>',
            unsafe_allow_html=True
        )

    # ── Toggle + New chat buttons ─────────────────────────────────────────────
    if st.session_state["retriever"] is not None:
        col1, col2, col3 = st.columns([4, 1.5, 1.5])

        with col2:
            label = "🙈 Hide upload" if st.session_state["show_upload"] else "📄 Change doc"
            if st.button(label, use_container_width=True):
                st.session_state["show_upload"] = not st.session_state["show_upload"]

        with col3:
            if st.button("🗑️ New chat", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

    # ── Upload section ────────────────────────────────────────────────────────
    if st.session_state["show_upload"]:
        uploaded_file = st.file_uploader(
            "Upload a PDF", type="pdf", label_visibility="collapsed"
        )

        if uploaded_file and uploaded_file.name != st.session_state["document_name"]:
            with st.spinner("Reading and indexing your document..."):
                try:
                    vectorstore, chunks = ingest(uploaded_file)
                    st.session_state["retriever"] = build_retriever(vectorstore, chunks)
                    st.session_state["document_name"] = uploaded_file.name
                    st.session_state["history"] = []
                    st.success(f"✅ Ready. Ask me anything about **{uploaded_file.name}**.")
                    st.session_state["show_upload"] = False
                    st.rerun()
                except ValueError as e:
                    st.error(f"⚠️ Document issue: {e}")
                except RuntimeError as e:
                    st.error(f"❌ Processing failed: {e}")

    # ── Chat history display ──────────────────────────────────────────────────
    # Replay the full history on every rerun so previous messages stay visible
    for message in st.session_state["history"]:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.write(message["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    question = st.chat_input("Ask a question about your document...")
    if question:
        if not st.session_state["retriever"]:
            st.warning("Please upload a document first.")
            return

        # Show the user message immediately
        with st.chat_message("user"):
            st.write(question)

        # Stream the assistant response
        with st.chat_message("assistant"):
            try:
                chunks = retrieve(question, st.session_state["retriever"])
            except RuntimeError as e:
                st.error(f"❌ Retrieval failed: {e}")
                return

            try:
                response = st.write_stream(
                    generate_stream(question, chunks, st.session_state["history"])
                )
            except RuntimeError as e:
                st.error(f"❌ Generation failed: {e}")
                return

        # Persist to history only after a successful response
        st.session_state["history"].append({"role": "user",  "content": question})
        st.session_state["history"].append({"role": "model", "content": response})


if __name__ == "__main__":
    main()