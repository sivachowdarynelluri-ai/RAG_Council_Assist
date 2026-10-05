"""
Council Assist - Evaluation
---------------------------
This reproduces the evaluation approach used in the Walert project:
  - NDCG@3 measures retrieval quality for answerable (Known/Inferred) questions
  - % correctly unanswered measures how often the system correctly refuses
    questions that are outside its knowledge base (Out-of-KB)

Run with:  python evaluate.py
"""

import json
import math
from rag import CouncilAssist


def ndcg_at_k(retrieved_ids, relevant_ids, k=3):
    """
    Normalised Discounted Cumulative Gain at rank k.
    Rewards putting relevant passages near the top of the results.
    A score of 1.0 means the relevant passages were ranked perfectly.
    """
    # DCG: add a score for each relevant passage, discounted by its position.
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1 / math.log2(i + 2)

    # IDCG: the best possible DCG, used to normalise the score to 0-1.
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


def main():
    assistant = CouncilAssist()

    with open("test_set.json") as f:
        test_questions = json.load(f)

    results = []
    for question in test_questions:
        response = assistant.ask(question["q"])
        retrieved_ids = [doc["id"] for doc, _ in response["sources"]]

        # Only answerable questions have relevance judgments for NDCG.
        if question["rel"]:
            ndcg = ndcg_at_k(retrieved_ids, question["rel"])
        else:
            ndcg = None

        results.append({
            **question,
            "top3": retrieved_ids,
            "top_score": round(response["sources"][0][1], 2),
            "answered": response["answered"],
            "ndcg3": ndcg,
        })

    # --- Summary by question type ---
    print("\n=== Evaluation Summary ===")
    for qtype in ["Known", "Inferred"]:
        group = [r for r in results if r["type"] == qtype]
        avg_ndcg = sum(r["ndcg3"] for r in group) / len(group)
        answered = sum(r["answered"] for r in group)
        print(f"{qtype}: n={len(group)}  NDCG@3={avg_ndcg:.3f}  answered={answered}/{len(group)}")

    out_of_kb = [r for r in results if r["type"] == "Out-of-KB"]
    correctly_refused = sum(not r["answered"] for r in out_of_kb)
    print(f"Out-of-KB: n={len(out_of_kb)}  correctly unanswered={correctly_refused}/{len(out_of_kb)}")

    # --- Per-question detail ---
    print("\n=== Per-question detail ===")
    for r in results:
        ndcg_text = f"{r['ndcg3']:.3f}" if r["ndcg3"] is not None else "  -  "
        print(f"{r['type'][:3]:<3} score={r['top_score']:<5} answered={str(r['answered']):<5} "
              f"ndcg={ndcg_text}  {r['q'][:60]}")

    # Save full results for the report.
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved detailed results to eval_results.json")


if __name__ == "__main__":
    main()