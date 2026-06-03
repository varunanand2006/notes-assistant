import sys
import os
import asyncio
import threading
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
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
async def ask(req: AskRequest):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run_query():
        try:
            index = get_index()
            engine = index.as_query_engine(streaming=True, similarity_top_k=4)
            streaming_response = engine.query(req.query)
            for token in streaming_response.response_gen:
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, f"\n[Error: {e}]")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=run_query, daemon=True).start()

    async def generate():
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()
