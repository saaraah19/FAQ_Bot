# app.py — orchestrator. Imports everything, calls everything. Nothing imports this.

import streamlit as st
from dotenv import load_dotenv
from google import genai
import os

from ingest import ingest
from retriever import build_retriever, retrieve
from generator import generate

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def main():
    st.title("FAQ Bot")

    # initialise session state on first run
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None

    # step 1 — document upload (runs once, result stored in session_state)
    uploaded_file = st.file_uploader("Upload a document", type="pdf")
    if uploaded_file and st.session_state["retriever"] is None:
        vectorstore, chunks = ingest(uploaded_file)
        st.session_state["retriever"] = build_retriever(vectorstore,chunks)
        st.success("Document ready. Ask a question.")

    # step 2 — chat interface
    question = st.chat_input("Ask a question about the document")
    if question and st.session_state["retriever"]:
        chunks = retrieve(question, st.session_state["retriever"])
        response = generate(question, chunks, st.session_state["history"])
        st.session_state["history"].append({"role": "user", "content": question})
        st.session_state["history"].append({"role": "model", "content": response})
        st.write(response)

if __name__ == "__main__":
    main()