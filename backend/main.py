import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, EMBED_MODEL, DOCS_DIR, STORAGE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SYSTEM_PROMPT

app = FastAPI()

Settings.llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=300.0, system_prompt=SYSTEM_PROMPT)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = CHUNK_OVERLAP

_index = None
ALLOWED_EXTS = {".pdf", ".md", ".txt"}


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
    seen = set()
    sources = []
    for node in response.source_nodes:
        name = os.path.basename(
            node.node.metadata.get("file_name") or node.node.metadata.get("file_path", "")
        )
        if name and name not in seen:
            seen.add(name)
            sources.append(name)
    return {"answer": str(response), "sources": sources}


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    docs_path = os.path.abspath(DOCS_DIR)
    os.makedirs(docs_path, exist_ok=True)

    saved_paths = []
    for file in files:
        name = os.path.basename(file.filename or "upload")
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTS:
            continue
        dest = os.path.join(docs_path, name)
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved_paths.append(dest)

    if not saved_paths:
        return {"error": "No valid files. Accepted: .pdf .md .txt"}

    try:
        new_docs = SimpleDirectoryReader(input_files=saved_paths).load_data()
    except Exception as e:
        return {"error": f"Failed to parse files: {e}"}

    index = get_index()
    for doc in new_docs:
        index.insert(doc)

    return {
        "added": len(new_docs),
        "files": [os.path.basename(p) for p in saved_paths],
    }


@app.post("/reingest")
def reingest():
    global _index
    docs_path = os.path.abspath(DOCS_DIR)
    storage_path = os.path.abspath(STORAGE_DIR)

    if not os.path.exists(docs_path) or not any(
        os.path.splitext(f)[1].lower() in ALLOWED_EXTS
        for f in os.listdir(docs_path)
    ):
        return {"status": "no_docs", "chunks": 0}

    client = chromadb.PersistentClient(path=storage_path)
    try:
        client.delete_collection("notes")
    except Exception:
        pass

    os.makedirs(storage_path, exist_ok=True)
    docs = SimpleDirectoryReader(
        docs_path,
        required_exts=list(ALLOWED_EXTS),
        recursive=True,
    ).load_data()

    collection = client.get_or_create_collection("notes")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    return {"status": "ok", "chunks": len(docs)}


@app.get("/files")
def list_files():
    docs_path = os.path.abspath(DOCS_DIR)
    if not os.path.exists(docs_path):
        return {"files": []}
    files = sorted(
        f for f in os.listdir(docs_path)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTS
    )
    return {"files": files}


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()
