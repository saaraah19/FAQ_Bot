# config.py — non-secret constants only
# API key goes in .env → loaded with os.getenv("GEMINI_API_KEY") in the file that needs it


EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
CHROMA_COLLECTION_NAME = "faq_docs"
CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
