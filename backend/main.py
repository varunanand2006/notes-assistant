import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, EMBED_MODEL, STORAGE_DIR

app = FastAPI()

Settings.llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=300.0)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

_index = None

def get_index():
    global _index
    if _index is None:
        chroma_client = chromadb.PersistentClient(path=os.path.abspath(STORAGE_DIR))
        chroma_collection = chroma_client.get_or_create_collection("notes")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return _index


class AskRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    index = get_index()
    engine = index.as_query_engine(similarity_top_k=4)
    response = engine.query(req.query)
    return {"answer": str(response)}


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()
