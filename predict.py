"""
Prediction + Evaluation Script
Uses the HybridRetriever (BM25 + FAISS + RRF) across both corpora.
"""
import pandas as pd
from retriever import get_retriever

def evaluate_f1(gold_lists, pred_lists):
    """Computes macro-averaged citation-level F1."""
    f1s = []
    for g_list, p_list in zip(gold_lists, pred_lists):
        g_set = set(g.strip() for g in g_list if g.strip())
        p_set = set(p.strip() for p in p_list if p.strip())
        
        if len(g_set) == 0 and len(p_set) == 0:
            f1s.append(1.0)
            continue
        if len(g_set) == 0 or len(p_set) == 0:
            f1s.append(0.0)
            continue
            
        tp = len(g_set & p_set)
        precision = tp / len(p_set)
        recall = tp / len(g_set)
        
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    
    return sum(f1s) / len(f1s) if f1s else 0.0

def run():
    print("Loading hybrid retriever...")
    retriever = get_retriever()
    
    # --- Validation ---
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)
    val_df = pd.read_csv("data/val.csv")
    val_queries = val_df['query'].tolist()
    gold_citations = [str(g).split(';') for g in val_df['gold_citations'].tolist()]
    
    # Try multiple K values to find the sweet spot
    for K in [5, 10, 15, 20, 30]:
        preds = retriever.retrieve(val_queries, top_k=K)
        f1 = evaluate_f1(gold_citations, preds)
        print(f"  K={K:3d}  →  F1 = {f1:.4f}")
    
    # Use K=15 for submission (seemed effective on train sets)
    SUBMIT_K = 15
    
    # --- Show detailed matches for debugging ---
    print("\n--- Detailed Val Results (K=15) ---")
    preds_15 = retriever.retrieve(val_queries, top_k=SUBMIT_K)
    for i, (gold, pred) in enumerate(zip(gold_citations, preds_15)):
        g_set = set(g.strip() for g in gold)
        p_set = set(p.strip() for p in pred)
        tp = g_set & p_set
        print(f"\nQuery {i}: {len(tp)}/{len(g_set)} gold found | {len(p_set)} predicted")
        if tp:
            print(f"  Matched: {list(tp)[:5]}")
    
    # --- Test Submission ---
    print("\n" + "="*60)
    print("GENERATING TEST SUBMISSION")
    print("="*60)
    test_df = pd.read_csv("data/test.csv")
    test_queries = test_df['query'].tolist()
    test_ids = test_df['query_id'].tolist()
    
    test_preds = retriever.retrieve(test_queries, top_k=SUBMIT_K)
    
    rows = []
    for q_id, preds in zip(test_ids, test_preds):
        rows.append({
            'query_id': q_id,
            'predicted_citations': ";".join(preds)
        })
    
    sub_df = pd.DataFrame(rows)
    sub_df.to_csv("submission.csv", index=False)
    print(f"✅ Saved submission.csv ({len(sub_df)} rows, K={SUBMIT_K})")

if __name__ == "__main__":
    run()
