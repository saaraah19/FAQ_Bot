"""his is the function responsible on !
        1.Loading the document and importing it
        2.wE'RE GOING TO CHUNK AND EMBED THE DOCUMENT THAN STORE IT IN CHROMA DB
so what do we need is :
1. embedding model to embed the chunks
2. pdf path to get the needed data to be embedded
3. store them in chromadb
"""
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from dotenv import load_dotenv
load_dotenv()

def load_document(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file not found.")
    if not pdf_path.endswith(".pdf"):
        raise ValueError("Expected .pdf file")
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def chunk_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

def ingest(uploaded_file):
    if uploaded_file.size == 0:
        raise ValueError("The uploaded file is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        documents = load_document(tmp_path)
        if not documents:
            raise ValueError("No text found in this PDF. It may be a scanned image.")

        chunks = chunk_text(documents)
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise ValueError("No usable content after chunking. Try a different document.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore, chunks

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to process document: {e}")
    finally:
        os.unlink(tmp_path)