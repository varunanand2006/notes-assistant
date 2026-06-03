import os

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "llama3.2:1b"
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
