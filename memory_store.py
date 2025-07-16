from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import os

embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME"))

def create_vector_store():
    return FAISS.from_texts(["_"], embedding)

def add_to_memory(store, summary: str, metadata: dict = None):
    store.add_texts([summary], [metadata] if metadata else None)

def retrieve_memory(store, query: str, k: int = 3):
    return store.similarity_search(query, k=k)
