import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from config import OLLAMA_BASE_URL, EMBED_MODEL, DOCS_DIR, STORAGE_DIR

def build_index():
    print("Loading documents from:", os.path.abspath(DOCS_DIR))

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    documents = SimpleDirectoryReader(
        os.path.abspath(DOCS_DIR),
        required_exts=[".pdf", ".md", ".txt"],
        recursive=True
    ).load_data()

    print(f"Loaded {len(documents)} document chunks")

    os.makedirs(os.path.abspath(STORAGE_DIR), exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=os.path.abspath(STORAGE_DIR))
    chroma_collection = chroma_client.get_or_create_collection("notes")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print("Index built and saved to storage/")
    return index

if __name__ == "__main__":
    build_index()