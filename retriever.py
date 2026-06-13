"""
okey so here the user asked a question about the document , now what we need to do actually is to :
1. Embed the query
2. Hybrid Search
3. Ensemble Retriever
4. Top-k Chunks
"""
from langchain.retrievers import EnsembleRetriever  # not langchain_classic
from langchain_community.retrievers import BM25Retriever
from config import TOP_K
def build_retriever(vectorstore,chunks):
    """Build an EnsembleRetriever combining BM25 + ChromaDB semantic search. Returns retriever."""
    semantic = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K

    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.5, 0.5]
    )

def retrieve(query, retriever):
    try:
        results = retriever.invoke(query)
        return results if results else []  # guard: never return None
    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {e}")