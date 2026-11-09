"""Central configuration. Tweak values here and re-run the evaluation to see the effect."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks

TOP_K_RETRIEVE = 20   # candidates pulled from each retriever before fusion
TOP_K_FINAL = 5       # chunks passed to the LLM
MAX_CONTEXT_CHUNKS = 8  # cap after merging sub-question results

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
