# 🧠 JARVIS — Swiss Legal Intelligence

> **Kaggle Competition**: [LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)

An AI-powered legal research assistant that retrieves relevant Swiss federal law provisions and court decisions using **hybrid vector + keyword search** across 2.4M+ legal documents.

## Architecture

| Component | Tech | Purpose |
|-----------|------|---------|
| Dense Retrieval | FAISS + MiniLM | Semantic cross-lingual search (EN→DE/FR/IT) |
| Sparse Retrieval | BM25 (rank_bm25) | Exact keyword matching |
| Score Fusion | Reciprocal Rank Fusion | Combines dense + sparse rankings |
| Backend | FastAPI + Uvicorn | REST API for retrieval |
| Frontend | Vanilla HTML/CSS/JS | Neon-themed conversational UI |

## Quick Start

```bash
# 1. Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download data (requires Kaggle CLI)
mkdir data && cd data
kaggle competitions download -c llm-agentic-legal-information-retrieval
unzip *.zip && rm *.zip && cd ..

# 3. Build indices (~30-60 min for full corpus)
python build_index.py

# 4. Run predictions
python predict.py

# 5. Launch Jarvis UI
python app.py
# Open http://localhost:8001
```

## Project Structure

```
├── build_index.py       # FAISS + BM25 index builder
├── retriever.py         # HybridRetriever (BM25 + FAISS + RRF)
├── predict.py           # Eval on val.csv + generate submission.csv
├── app.py               # FastAPI backend + Jarvis persona
├── static_app/          # Neon-themed frontend
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── requirements.txt
└── data/                # Kaggle dataset (gitignored)
```

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
