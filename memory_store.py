import os
from dotenv import load_dotenv

load_dotenv()

# LangChain moved most integrations to the `langchain_community` umbrella. Using the
# new import paths eliminates deprecation warnings and keeps the codebase future-proof.
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
from datetime import datetime
from typing import Dict, Optional

# Global dictionary to store vector stores by topic
vector_stores: Dict[str, FAISS] = {}

embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text"))

VECTOR_STORE_DIR = "vector_stores"

def create_vector_store():
    """Create a new empty vector store"""
    return FAISS.from_texts(["dummy"], embedding)

def save_vector_store(store, path: str):
    """Save the vector store to local disk"""
    # Use LangChain's safer JSON serialization format so that future loads do not
    # require un-pickling arbitrary data.
    store.save_local(path, safe_serialization=True)

def load_vector_store(path: str):
    """Load vector store from local disk"""
    # LangChain's `load_local` requires `allow_dangerous_deserialization=True` when un-pickling
    # vector stores.  We want to enable this *only* for data that originates from the
    # current application (i.e. the folder managed by `VECTOR_STORE_DIR`).  This avoids
    # blindly un-pickling files from arbitrary, potentially malicious sources.

    # A path is considered trusted when it lives inside our configured VECTOR_STORE_DIR.
    def _is_trusted_path(p: str) -> bool:
        """Return True when *p* is located inside the VECTOR_STORE_DIR folder."""
        abs_base = os.path.abspath(VECTOR_STORE_DIR)
        abs_path = os.path.abspath(p)
        try:
            # commonpath will raise ValueError for paths on different drives (Windows)
            return os.path.commonpath([abs_base, abs_path]) == abs_base
        except ValueError:
            return False

    allow_pickle = _is_trusted_path(path)

    try:
        return FAISS.load_local(path, embedding, allow_dangerous_deserialization=allow_pickle)
    except Exception as e:
        print(
            f"Could not load vector store from {path}: {e}\n"
            "If this is a trusted path you can delete the corrupted store and it will "
            "be recreated automatically."
        )
        return None

def get_or_create_vector_store(topic: str = "default") -> FAISS:
    """Get or create a vector store for a specific topic"""
    if topic in vector_stores:
        return vector_stores[topic]

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    path = os.path.join(VECTOR_STORE_DIR, f"{topic}_vector_store")
    store = load_vector_store(path)
    if store is None:
        store = create_vector_store()
        save_vector_store(store, path)

    # Cache the store
    vector_stores[topic] = store
    return store

def get_available_topics() -> list[str]:
    """Get list of available vector store topics"""
    topics = []
    for file in os.listdir():
        if file.startswith("vector_store_") and os.path.isdir(file):
            topics.append(file.replace("vector_store_", ""))
    return topics or ["general"]  # Return at least "general" topic

def add_to_memory(store, summary: str, metadata: dict = None):
    """Add summary to memory with enhanced metadata"""
    doc = Document(page_content=summary, metadata=metadata or {})
    store.add_documents([doc])

def retrieve_memory(store, query: str, k: int = 3):
    """Retrieve memory with semantic search"""
    return store.similarity_search(query, k=k)

def retrieve_memory_by_topic(topic: str, query: str = "", k: int = 3):
    """Retrieve memory from specific topic store with optional query"""
    store = get_or_create_vector_store(topic)
    if query:
        return retrieve_memory(store, query, k)
    return store.similarity_search("", k=k)  # Return random docs if no query

def keyword_search(store, query: str, k: int = 3):
    """Simple keyword search in document contents"""
    all_docs = store.similarity_search("", k=1000)  # Get all available docs
    matches = [doc for doc in all_docs if query.lower() in doc.page_content.lower()]
    return matches[:k]

def hybrid_search(store, query: str, k: int = 3):
    """Combine semantic and keyword search results"""
    semantic_results = retrieve_memory(store, query, k)
    keyword_results = keyword_search(store, query, k)

    # Combine results, prioritizing docs found by both methods
    seen = set()
    combined = []
    for doc in semantic_results + keyword_results:
        key = doc.page_content
        if key not in seen:
            combined.append(doc)
            seen.add(key)
    return combined[:k]