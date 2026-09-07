"""Flat BM25 retrieval baseline over the raw AlzKG sentence corpus.

Answers a question the Graph-RAG ablation (retrieval_ablation.py) cannot on
its own: does the knowledge-graph structure itself add anything over simply
retrieving relevant sentences from the same corpus by lexical overlap? Uses
the identical probe set and the identical gold-coverage criterion (>=50% of
gold-answer tokens present in the retrieved evidence text) as
retrieval_ablation.py, so the two numbers are directly comparable -- with
one important asymmetry noted in the paper (Sec. 3.2): BM25 returns raw
sentence text, which is more likely to contain the gold phrase verbatim,
while Graph-RAG returns compressed entity-relation paths, not full sentences.

Pure stdlib; no new dependency on the project's runtime.
"""

import argparse
import math
import time
from collections import defaultdict

from alzgraph.common import read_json, read_jsonl, write_json
from alzgraph.metrics import _tokens


def load_probes() -> list[dict]:
    probes = []
    for item in read_json("data/alzbench/t1/mcq.json"):
        probes.append({"query": item["question"], "gold": item.get("answer", "")})
    for item in read_json("data/alzbench/t3/bpm_mcq.json"):
        probes.append({"query": item["clinical_scenario"], "gold": item.get("recommended", "")})
    return probes


def covered(gold: str, evidence_tokens: set) -> bool:
    gold_tokens = [t for t in _tokens(gold) if len(t) > 2]
    if not gold_tokens:
        return False
    hits = sum(1 for t in gold_tokens if t in evidence_tokens)
    return hits / len(gold_tokens) >= 0.5


def build_index(sentences_path: str):
    sents = [s["text"] for rec in read_jsonl(sentences_path) for s in rec["candidate_sentences"]]
    doc_tokens = [_tokens(s) for s in sents]
    doc_len = [len(dt) for dt in doc_tokens]
    avgdl = sum(doc_len) / max(len(sents), 1)
    inverted: dict = defaultdict(dict)
    for i, dt in enumerate(doc_tokens):
        tf: dict = {}
        for tok in dt:
            tf[tok] = tf.get(tok, 0) + 1
        for tok, c in tf.items():
            inverted[tok][i] = c
    return sents, inverted, doc_len, avgdl


def bm25_search(query: str, sents, inverted, doc_len, avgdl, n_docs: int, top_k: int = 12, k1: float = 1.5, b: float = 0.75):
    qtoks = [t for t in set(_tokens(query)) if t in inverted]
    scores: dict = defaultdict(float)
    for tok in qtoks:
        postings = inverted[tok]
        idf = math.log((n_docs - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
        for doc_id, tf in postings.items():
            denom = tf + k1 * (1 - b + b * doc_len[doc_id] / avgdl)
            scores[doc_id] += idf * (tf * (k1 + 1)) / denom
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [sents[i] for i, _ in ranked]


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 flat-retrieval baseline over the AlzKG sentence corpus.")
    parser.add_argument("--sentences", default="data/corpus/fulltext_sentences.jsonl")
    parser.add_argument("--top-k", type=int, default=12, help="Matches Graph-RAG's max_paths budget.")
    parser.add_argument("--out", default="data/alzkg/bm25_baseline.json")
    args = parser.parse_args()

    sents, inverted, doc_len, avgdl = build_index(args.sentences)
    n_docs = len(sents)

    probes = load_probes()
    n_cov, latencies = 0, []
    for p in probes:
        t0 = time.perf_counter()
        top = bm25_search(p["query"], sents, inverted, doc_len, avgdl, n_docs, top_k=args.top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        evidence_tokens = {t for s in top for t in _tokens(s)}
        n_cov += int(covered(p["gold"], evidence_tokens))

    result = {
        "method": f"BM25 flat sentence retrieval, top-{args.top_k}, same corpus as Graph-RAG",
        "n_probes": len(probes),
        "gold_coverage": round(n_cov / len(probes), 3),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
    }
    write_json(result, args.out)
    print(result)


if __name__ == "__main__":
    main()
