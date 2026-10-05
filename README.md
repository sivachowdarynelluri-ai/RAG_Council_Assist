# Council Assist – prototype (WIL Group 81)
**Group ID:** WIL Project 81

**Team members:**
- Mohammed Sufiyan (S4191466)
- Likith Vudugula (S4203933)
- Siva Nelluri (S4158617)
- Qadar Mohiuddin Syed (S4203189)
- Mohammed Faizan Uddin (S4203995)

Test-driven RAG assistant for City of Melbourne resident FAQs.
BM25 retrieval (written from scratch) + local LLM via Ollama + refusal when the answer isn't in the knowledge base.

## Run it

Test-driven RAG assistant for City of Melbourne resident FAQs.
BM25 retrieval (written from scratch) + local LLM via Ollama + refusal when the answer isn't in the knowledge base.

## Run it
```
pip install streamlit
ollama pull llama3.2:3b        # optional – without Ollama the app uses extractive answers
streamlit run app.py or python -m streamlit run app.py
```
Change the model with `OLLAMA_MODEL=<name> streamlit run app.py` (use whatever model the team used).
Change the refusal threshold with `CA_THRESHOLD=4.0`.

## Files
- `kb.json` – knowledge base (18 passages, each with its source URL). Swap in the team's passages if preferred.
- `rag.py` – BM25 retriever, Ollama generator, refusal logic
- `app.py` – Streamlit chat interface (answers, sources, relevance scores)
- `test_set.json` – 18 test questions: 8 known, 5 inferred, 5 out-of-knowledge-base
- `evaluate.py` – Walert-style evaluation: NDCG@3 and % correctly unanswered

## Results (retrieval + refusal, `python evaluate.py`)
| Question type | n | Metric | Result |
|---|---|---|---|
| Known | 8 | NDCG@3 | 0.938 |
| Inferred | 5 | NDCG@3 | 0.676 |
| Out-of-KB | 5 | % correctly unanswered | 80% (4/5) |

# About the project

Council Assist answers residents' questions about City of Melbourne services (bins, permits, rates, pet registration, and more) by retrieving relevant passages from a curated knowledge base and generating a plain-language answer grounded in them. When no relevant passage is found, the system refuses to answer rather than guessing — a key safety feature for civic information, following the evaluation approach of the Walert project (Pathiyan Cherumanal et al., 2024).

# Known limitation

BM25 matches keywords rather than meaning, so semantically related terms (e.g. "kitten" vs "cat") may not match. Planned future work is to add embedding-based retrieval to capture meaning, alongside expanding the knowledge base and adding a faithfulness metric.