"""
Configuration file for RAG Chatbot project.
Centralized paths and settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
CHUNKS_DIR = DATA_DIR / "chunks"

# Database paths
RAG_DB_PATH = str(DB_DIR / "rag_chatbot.db")
CHAT_HISTORY_DB_PATH = str(DB_DIR / "chat_history.db")

# Index directories
INDEX_DIR = PROJECT_ROOT / "index"
BM25_INDEX_PATH = str(INDEX_DIR / "bm25_index.pkl")
FAISS_INDEX_PATH = str(INDEX_DIR / "faiss_index.bin")

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
HF_MODELS_DIR = PROJECT_ROOT / "hf_models"
BARTPHO_ADAPTER_DIR = str(MODELS_DIR / "bartpho_lora_adapter")

# Chunks paths
CHUNKS_NEW_PATH = str(CHUNKS_DIR / "chunks_new.json")
CHUNKS_TABLE_AWARE_PATH = str(CHUNKS_DIR / "chunks_table_aware.json")
CHUNKS_TABLE_ONLY_PATH = str(CHUNKS_DIR / "chunks_table_only.json")

# Environment variables (with defaults)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "local-admin-token")
MAX_CONCURRENT_GENERATIONS = int(os.environ.get("MAX_CONCURRENT_GENERATIONS", "1"))

# Ensure directories exist
DB_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
