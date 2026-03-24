"""
build_index.py — FAISS + BM25 index builder for Swiss legal corpora.

Indexes:
  1. laws_de.csv       (~175K articles)
  2. court_considerations.csv (~2.4M decisions)

Usage:
  python build_index.py
"""

import os
import re
import pickle

import faiss
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODELS_DIR = "models"
DATA_DIR = "data"

CORPORA = [
    {
        "name": "laws",
        "csv": f"{DATA_DIR}/laws_de.csv",
        "chunksize": 50_000,
    },
    {
        "name": "court",
        "csv": f"{DATA_DIR}/court_considerations.csv",
        "chunksize": 50_000,
    },
]

# ── Device ────────────────────────────────────────────────

def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _select_device()

# ── Helpers ───────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Whitespace-and-punctuation tokenizer for BM25."""
    return re.findall(r"\w+", str(text).lower())


def load_model() -> SentenceTransformer:
    print(f"[model]  {MODEL_NAME}  on  {DEVICE}")
    return SentenceTransformer(MODEL_NAME, device=DEVICE)


# ── Index Builder ─────────────────────────────────────────

def build(
    csv_path: str,
    name: str,
    model: SentenceTransformer,
    chunksize: int = 50_000,
    text_col: str = "text",
    id_col: str = "citation",
) -> None:
    """Build a FAISS dense index and a BM25 sparse index for one corpus."""

    faiss_path   = f"{MODELS_DIR}/{name}.index"
    mapping_path = f"{MODELS_DIR}/{name}_mapping.pkl"
    bm25_path    = f"{MODELS_DIR}/{name}_bm25.pkl"

    print(f"\n{'─'*50}")
    print(f"  Indexing  {csv_path}  →  {name}")
    print(f"{'─'*50}")

    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    id_map: list[str] = []
    bm25_tokens: list[list[str]] = []

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        chunk = chunk.dropna(subset=[text_col, id_col])
        texts = chunk[text_col].tolist()
        ids = chunk[id_col].tolist()
        if not texts:
            continue

        print(f"  chunk {i + 1}  ({len(texts):,} texts)")

        embs = model.encode(
            texts,
            batch_size=512,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        index.add(np.asarray(embs))
        id_map.extend(ids)
        bm25_tokens.extend(tokenize(t) for t in texts)

    # Persist
    faiss.write_index(index, faiss_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(id_map, f)

    print(f"  BM25 fitting {len(bm25_tokens):,} docs …")
    bm25 = BM25Okapi(bm25_tokens)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"  ✅  {name}: {index.ntotal:,} vectors saved\n")


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    m = load_model()

    for corpus in CORPORA:
        build(corpus["csv"], corpus["name"], m, chunksize=corpus["chunksize"])

    print("All indices built.")
