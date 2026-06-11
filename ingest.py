"""his is the function responsible on !
        1.Loading the document and importing it
        2.wE'RE GOING TO CHUNK AND EMBED THE DOCUMENT THAN STORE IT IN CHROMA DB
so what do we need is :
1. embedding model to embed the chunks
2. pdf path to get the needed data to be embedded
3. store them in chromadb
"""
from langchain_community.vectorstores import Chroma

from config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, EMBEDDING_MODEL
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()

def load_document(pdf_path):
    """Load a PDF file. Returns a list of LangChain Document objects."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file not found.")
    if not pdf_path.endswith(".pdf"):
        raise ValueError("Expected .pdf file")

    loader = PyPDFLoader(pdf_path)
    return loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
def chunk_text(documents):
    """Split Document objects into smaller overlapping chunks. Returns list of chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def ingest(uploaded_file):
    """Orchestrates: load → chunk → embed → store in ChromaDB. Returns the vectorstore."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        documents = load_document(tmp_path)
        chunks = chunk_text(documents)
        # now embed chunks and store in ChromaDB ← this is what's left

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_DB_PATH
        )
        return vectorstore,chunks
    finally:
        os.unlink(tmp_path)
