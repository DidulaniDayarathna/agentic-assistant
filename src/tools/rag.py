"""
Document search (RAG) tool.

Indexes everything under `settings.rag_data_dir` into a local FAISS
vector store using a small sentence-transformers embedding model (runs
on CPU, no API key required), then exposes a `search_documents` tool
the agent can call for grounded, citeable answers instead of relying
on the LLM's parametric memory.

The index is built lazily on first use and cached in-process, so
importing this module has no startup cost — useful for tests and for
keeping app startup fast.
"""
import logging
from functools import lru_cache
from pathlib import Path

from langchain.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache
def _get_vectorstore() -> FAISS | None:
    data_dir = Path(settings.rag_data_dir)
    if not data_dir.exists() or not any(data_dir.glob("*.txt")):
        logger.warning("RAG data dir %s has no .txt files; document search disabled.", data_dir)
        return None

    loader = DirectoryLoader(str(data_dir), glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=settings.rag_embedding_model)
    logger.info("Building FAISS index from %d chunks (%d source docs).", len(chunks), len(docs))
    return FAISS.from_documents(chunks, embeddings)


@tool
def search_documents(query: str) -> str:
    """Search internal company/product documents for an answer.

    Use this for questions about internal policy, product specs, or
    anything that should be grounded in the provided document set
    rather than general web knowledge.

    Args:
        query: The question or topic to search the documents for.
    """
    store = _get_vectorstore()
    if store is None:
        return "Document search is not available: no documents are indexed."

    hits = store.similarity_search(query, k=3)
    if not hits:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(hits, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        formatted.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)
