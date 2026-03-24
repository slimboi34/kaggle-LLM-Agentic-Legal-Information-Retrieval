"""
Hybrid Retriever: BM25 + FAISS Dense + Reciprocal Rank Fusion
Searches both laws and court decisions simultaneously.
"""
import pandas as pd
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pickle
import os
import re

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(MODEL_NAME, device=device)


def tokenize_for_bm25(text):
    return re.findall(r'\w+', str(text).lower())


class CorpusRetriever:
    """Handles retrieval from a single corpus (laws OR court decisions)."""
    
    def __init__(self, faiss_path, mapping_path, bm25_path=None):
        self.index = faiss.read_index(faiss_path)
        with open(mapping_path, 'rb') as f:
            self.mapping = pickle.load(f)
        self.bm25 = None
        if bm25_path and os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
    
    def dense_search(self, query_embedding, top_k=50):
        """Returns list of (citation, score) tuples from FAISS."""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        results = []
        for j in range(top_k):
            idx = indices[0][j]
            if idx != -1 and idx < len(self.mapping):
                results.append((self.mapping[idx], float(distances[0][j])))
        return results
    
    def bm25_search(self, query_tokens, top_k=50):
        """Returns list of (citation, score) tuples from BM25."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0 and idx < len(self.mapping):
                results.append((self.mapping[idx], float(scores[idx])))
        return results


class HybridRetriever:
    """
    Combines BM25 + FAISS across multiple corpora using Reciprocal Rank Fusion.
    """
    
    def __init__(self):
        self.corpora = []
        
    def add_corpus(self, name, faiss_path, mapping_path, bm25_path=None):
        if os.path.exists(faiss_path) and os.path.exists(mapping_path):
            print(f"Loading corpus: {name}")
            retriever = CorpusRetriever(faiss_path, mapping_path, bm25_path)
            self.corpora.append((name, retriever))
            print(f"  ✅ {name}: {retriever.index.ntotal} vectors")
        else:
            print(f"  ⚠️  Skipping {name}: index files not found")
    
    def retrieve(self, queries, top_k=10, dense_k=100, bm25_k=100, rrf_k=60):
        """
        Retrieve citations for a list of queries using hybrid search.
        
        Args:
            queries: list of query strings
            top_k: number of final citations to return per query
            dense_k: candidates from dense search per corpus
            bm25_k: candidates from BM25 search per corpus
            rrf_k: RRF constant (higher = less emphasis on rank position)
        
        Returns: list of list of citation strings
        """
        # Encode all queries at once
        embeddings = model.encode(
            queries, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        all_results = []
        
        for i, query in enumerate(queries):
            query_tokens = tokenize_for_bm25(query)
            query_emb = embeddings[i]
            
            # Collect candidates from all corpora
            rrf_scores = {}
            
            for corpus_name, retriever in self.corpora:
                # Dense retrieval
                dense_results = retriever.dense_search(query_emb, top_k=dense_k)
                for rank, (citation, score) in enumerate(dense_results):
                    if citation not in rrf_scores:
                        rrf_scores[citation] = 0.0
                    rrf_scores[citation] += 1.0 / (rrf_k + rank + 1)
                
                # BM25 retrieval
                bm25_results = retriever.bm25_search(query_tokens, top_k=bm25_k)
                for rank, (citation, score) in enumerate(bm25_results):
                    if citation not in rrf_scores:
                        rrf_scores[citation] = 0.0
                    rrf_scores[citation] += 1.0 / (rrf_k + rank + 1)
            
            # Sort by RRF score and take top_k
            sorted_citations = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            all_results.append([cit for cit, score in sorted_citations[:top_k]])
        
        return all_results


def get_retriever():
    """Factory function to create a fully loaded HybridRetriever."""
    retriever = HybridRetriever()
    retriever.add_corpus(
        "laws",
        "models/laws.index", "models/laws_mapping.pkl", "models/laws_bm25.pkl"
    )
    retriever.add_corpus(
        "court",
        "models/court.index", "models/court_mapping.pkl", "models/court_bm25.pkl"
    )
    return retriever


if __name__ == "__main__":
    r = get_retriever()
    results = r.retrieve(["Can someone work in Switzerland without a permit?"], top_k=5)
    print("Test results:", results)
