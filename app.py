# app.py — orchestrator. Imports everything, calls everything. Nothing imports this.

import os

import streamlit as st
from dotenv import load_dotenv

from ingest import ingest
from retriever import build_retriever, retrieve
from generator import generate

load_dotenv()
# ── Page configuration ────────────────────────────────────────────────────────
# Must be the first Streamlit call. Sets the browser tab title and icon.
st.set_page_config(page_title="FAQ Bot", page_icon="💬", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
# Streamlit lets you inject raw CSS with st.markdown + unsafe_allow_html.
# We use it here to style the background, the header card, and the chat bubbles.
st.markdown("""
<style>
    .stApp { background-color: #f4f6f9; }

    /* The purple gradient header card at the top of the page */
    .bot-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .bot-header h1 { margin: 0; font-size: 1.6rem; font-weight: 600; }
    .bot-header p  { margin: 0.2rem 0 0; font-size: 0.9rem; opacity: 0.85; }

    /* Small blue pill that shows the loaded document name */
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

    /* Add a white card + subtle shadow behind each chat message */
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
# Streamlit reruns app.py from top to bottom on every user interaction.
# st.session_state is the only thing that survives between reruns.
# We initialise all keys once here — if the key already exists, we skip it.
def init_state():
    defaults = {
        "history": [],          # list of {role, content} dicts — the conversation
        "retriever": None,      # the EnsembleRetriever built after upload
        "document_name": None,  # tracks which file is loaded (prevents re-ingesting same file)
        "show_upload": True,    # controls whether the upload widget is visible
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

    # ── Toggle + New chat buttons ─────────────────────────────────────────────
    # Only appear once a document is loaded — no point showing them before that.
    if st.session_state["retriever"] is not None:
        col1, col2, col3 = st.columns([4, 1.5, 1.5])

        with col2:
            # Button label changes depending on current visibility state
            label = "🙈 Hide upload" if st.session_state["show_upload"] else "📄 Change doc"
            if st.button(label, use_container_width=True):
                # Flip the boolean — True becomes False, False becomes True
                st.session_state["show_upload"] = not st.session_state["show_upload"]

        with col3:
            if st.button("🗑️ New chat", use_container_width=True):
                st.session_state["history"] = []  # wipe conversation history
                st.rerun()  # force immediate re-render so chat clears visually

    # ── Upload section ────────────────────────────────────────────────────────
    # Only rendered when show_upload is True.

 # ── Upload section ────────────────────────────────────────────────────────
    if st.session_state["show_upload"]:
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

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
                    # clean validation error — we wrote this message ourselves
                    st.error(f"⚠️ Document issue: {e}")
                except RuntimeError as e:
                    # unexpected failure from API or file system
                    st.error(f"❌ Processing failed: {e}")

    # ── Chat input ────────────────────────────────────────────────────────────
    question = st.chat_input("Ask a question about your document...")
    if question:
        if not st.session_state["retriever"]:
            st.warning("Please upload a document first.")
            return

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chunks = retrieve(question, st.session_state["retriever"])
                    response = generate(question, chunks, st.session_state["history"])
                except RuntimeError as e:
                    st.error(f"❌ {e}")
                    return  # stop here — don't append a failed response to history
            st.write(response)

        st.session_state["history"].append({"role": "user", "content": question})
        st.session_state["history"].append({"role": "model", "content": response})


if __name__ == "__main__":
    main()
