"""
Council Assist - RAG pipeline
------------------------------
This file contains the core logic for our council FAQ assistant:
  1. BM25 retrieval (written from scratch) to find the most relevant passages
  2. Answer generation using a local Ollama model
  3. A refusal rule so the system says "I don't know" instead of guessing

We built BM25 ourselves rather than using a library, to reproduce the
retrieval approach used in the Walert project.
"""

import json
import math
import re
import os
import urllib.request
from collections import Counter


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

# Common words that carry little meaning. We remove these before matching so
# that scoring focuses on the important words in a question.
STOP_WORDS = set(
    "a an the of to in on for and or is are be can i my do does how what when "
    "where which who with by at it as this that from your you me we our if not "
    "no".split()
)


def simple_stem(word):
    """
    A very light stemmer so that plural words match their singular form
    (e.g. "bins" -> "bin", "batteries" -> "battery", "mattresses" -> "mattress").
    This is intentionally simple - it only handles common English endings, and
    is careful not to over-trim words like "rates" (which should become "rate",
    not "rat").
    """
    if len(word) < 4:
        return word

    if word.endswith("sses"):            # mattresses -> mattress
        return word[:-2]
    if word.endswith("ies"):             # batteries -> battery
        return word[:-3] + "y"
    if word.endswith("es"):
        # For words ending in x/s/z or ch/sh, drop the whole "es" (boxes -> box).
        # Otherwise only drop the "s" so we keep the final "e" (rates -> rate).
        if word[-3] in "xsz" or word.endswith(("ches", "shes")):
            return word[:-2]
        return word[:-1]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]                 # bins -> bin
    return word


def tokenize(text):
    """Turn a piece of text into a clean list of keywords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [simple_stem(w) for w in words if w not in STOP_WORDS]


# ---------------------------------------------------------------------------
# BM25 retriever (built from scratch)
# ---------------------------------------------------------------------------

class BM25:
    """
    Okapi BM25 ranking, implemented from scratch.
    BM25 scores how relevant each passage is to a query based on shared words,
    rewarding rare words and adjusting for passage length.
    k1 and b are the standard BM25 tuning constants.
    """

    def __init__(self, documents, k1=1.2, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b

        # Tokenise every passage (title + text) once, up front.
        self.doc_tokens = [
            tokenize(doc["title"] + " " + doc["text"]) for doc in documents
        ]
        self.num_docs = len(documents)
        self.avg_doc_length = sum(len(t) for t in self.doc_tokens) / self.num_docs

        # Count how many passages each word appears in (document frequency).
        doc_freq = Counter()
        for tokens in self.doc_tokens:
            for word in set(tokens):
                doc_freq[word] += 1

        # Inverse document frequency: rarer words get a higher weight.
        self.idf = {
            word: math.log(1 + (self.num_docs - freq + 0.5) / (freq + 0.5))
            for word, freq in doc_freq.items()
        }

        # Pre-count word frequencies within each passage.
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

    def score_all(self, query):
        """Return a BM25 score for every passage against the query."""
        query_words = tokenize(query)
        scores = []

        for i, term_freq in enumerate(self.term_freqs):
            doc_length = len(self.doc_tokens[i])
            score = 0.0
            for word in query_words:
                if word in term_freq:
                    freq = term_freq[word]
                    numerator = self.idf[word] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (
                        1 - self.b + self.b * doc_length / self.avg_doc_length
                    )
                    score += numerator / denominator
            scores.append(score)
        return scores

    def search(self, query, k=3):
        """Return the top-k passages and their scores, best first."""
        scores = self.score_all(query)
        ranked_indexes = sorted(range(self.num_docs), key=lambda i: -scores[i])[:k]
        return [(self.documents[i], scores[i]) for i in ranked_indexes]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

REFUSAL_MESSAGE = (
    "Sorry, I don't have that information in the council knowledge base. "
    "Please contact the City of Melbourne on 03 9658 9658."
)

# These can be changed with environment variables if needed.
THRESHOLD = float(os.environ.get("CA_THRESHOLD", "4.0"))      # min BM25 score to answer
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

PROMPT_TEMPLATE = """You are Council Assist, a helpful assistant for City of Melbourne residents.
Answer the question using ONLY the passages below. Keep it to 2-3 sentences.
If the passages do not contain the answer, reply exactly: "I don't know."

Passages:
{context}

Question: {question}
Answer:"""


def ollama_available():
    """Check whether a local Ollama server is running."""
    try:
        urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=1)
        return True
    except Exception:
        return False


def generate_with_ollama(question, passages):
    """Ask the local Ollama model to answer using the retrieved passages."""
    context = "\n".join(f"[{doc['id']}] {doc['text']}" for doc, _ in passages)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    request_body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},   # 0 = deterministic, no random variation
    }).encode()

    request = urllib.request.Request(
        OLLAMA_URL + "/api/generate",
        request_body,
        {"Content-Type": "application/json"},
    )
    response = urllib.request.urlopen(request, timeout=120)
    return json.loads(response.read())["response"].strip()


def generate_extractive(question, passages):
    """
    Fallback used when Ollama isn't running.
    Instead of generating a sentence, we pick the sentences from the top
    passages that share the most words with the question.
    """
    question_words = set(tokenize(question))
    top_score = passages[0][1]
    scored_sentences = []

    for rank, (doc, passage_score) in enumerate(passages):
        # Only use passages that are nearly as relevant as the best one.
        if passage_score < 0.5 * top_score:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", doc["text"]):
            overlap = len(question_words & set(tokenize(sentence)))
            scored_sentences.append((overlap * passage_score, rank, sentence))

    # Keep the two best-matching sentences, then restore reading order.
    best = [s for s in sorted(scored_sentences, key=lambda x: -x[0]) if s[0] > 0][:2]
    best.sort(key=lambda x: x[1])

    if best:
        return " ".join(sentence for _, _, sentence in best)
    return passages[0][0]["text"]   # last resort: just return the top passage


# ---------------------------------------------------------------------------
# Main assistant class
# ---------------------------------------------------------------------------

class CouncilAssist:
    """Ties retrieval, generation, and the refusal rule together."""

    def __init__(self, kb_path=None):
        if kb_path is None:
            kb_path = os.path.join(os.path.dirname(__file__), "kb.json")
        with open(kb_path) as f:
            self.kb = json.load(f)
        self.bm25 = BM25(self.kb)

    def ask(self, question, k=3):
        """
        Answer a question. Returns a dictionary with the answer, whether we
        answered or refused, the sources used, and which mode produced it.
        """
        hits = self.bm25.search(question, k)
        top_score = hits[0][1]

        # Refusal rule 1: nothing retrieved was relevant enough.
        if top_score < THRESHOLD:
            return {
                "answer": REFUSAL_MESSAGE,
                "answered": False,
                "sources": hits,
                "mode": "refused (low retrieval score)",
            }

        # If Ollama is running, let it write the answer.
        if ollama_available():
            answer = generate_with_ollama(question, hits)
            mode = f"Ollama ({OLLAMA_MODEL})"
            # Refusal rule 2: the model itself says it can't answer.
            if "i don't know" in answer.lower():
                return {
                    "answer": REFUSAL_MESSAGE,
                    "answered": False,
                    "sources": hits,
                    "mode": mode + " - declined",
                }
        else:
            # Otherwise use the extractive fallback.
            answer = generate_extractive(question, hits)
            mode = "extractive mode (Ollama not running)"

        return {
            "answer": answer,
            "answered": True,
            "sources": hits,
            "mode": mode,
        }