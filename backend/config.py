import os

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
