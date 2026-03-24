"""
JARVIS Legal AI — FastAPI Backend
Serves the retrieval API + static frontend.
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from contextlib import asynccontextmanager

retriever = None

@asynccontextmanager
async def lifespan(app):
    """Load the retriever on startup."""
    global retriever
    from retriever import get_retriever
    print("Loading JARVIS retrieval engine...")
    retriever = get_retriever()
    print("✅ JARVIS is online.")
    yield

app = FastAPI(title="JARVIS Legal AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class QueryResponse(BaseModel):
    citations: List[str]
    analysis: str


# ── Jarvis Persona Synthesis ─────────────────────────────
def synthesize_analysis(query: str, citations: List[str]) -> str:
    """
    Generates a Jarvis-style legal analysis based on the retrieved citations.
    Uses rule-based synthesis (no external LLM API needed).
    """
    law_cites = [c for c in citations if c.startswith('Art.')]
    court_cites = [c for c in citations if c.startswith('BGE') or '/' in c]
    
    parts = []
    
    if law_cites and court_cites:
        parts.append(
            f"I've identified {len(law_cites)} relevant statutory provision{'s' if len(law_cites) != 1 else ''} "
            f"and {len(court_cites)} pertinent Federal Court decision{'s' if len(court_cites) != 1 else ''} "
            f"that bear directly on your inquiry."
        )
    elif law_cites:
        parts.append(
            f"I've located {len(law_cites)} statutory provision{'s' if len(law_cites) != 1 else ''} "
            f"of direct relevance to your question."
        )
    elif court_cites:
        parts.append(
            f"I've identified {len(court_cites)} Federal Court decision{'s' if len(court_cites) != 1 else ''} "
            f"that address the legal issues you've raised."
        )
    
    if law_cites:
        primary = law_cites[0]
        parts.append(
            f"\nThe primary statutory basis appears to be {primary}. "
            f"I recommend analyzing this provision carefully in conjunction with the supplementary authorities below."
        )
    
    if court_cites:
        primary_court = court_cites[0]
        if primary_court.startswith('BGE'):
            parts.append(
                f"\nFrom the jurisprudence, {primary_court} establishes the leading precedent. "
                f"This decision and its reasoning chain should anchor your legal argument."
            )
        else:
            parts.append(
                f"\nThe decision {primary_court} provides relevant case-specific guidance. "
                f"While not a leading BGE decision, its reasoning may prove instructive."
            )
    
    parts.append(
        "\nI'd recommend building your argument with these authorities as the foundation. "
        "Each citation below links to a specific provision or court consideration in the Swiss federal corpus."
    )
    
    return "\n".join(parts)


@app.post("/api/query", response_model=QueryResponse)
def query_legal_database(request: QueryRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="JARVIS is still initializing")
    
    try:
        results = retriever.retrieve([request.query], top_k=request.top_k)
        citations = results[0] if results else []
        analysis = synthesize_analysis(request.query, citations)
        return QueryResponse(citations=citations, analysis=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files LAST (catch-all)
app.mount("/", StaticFiles(directory="static_app", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
