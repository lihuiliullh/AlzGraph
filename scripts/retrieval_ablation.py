"""Intrinsic Graph-RAG retrieval ablation over AlzKG (no LLM required).

Measures, as a function of the retrieval budget (max subgraph nodes) and the
maximum reasoning-path depth:
  - the average number of serialized reasoning paths returned,
  - gold-evidence coverage: the fraction of probe queries whose gold answer
    concept is recovered inside the retrieved paths,
  - mean retrieval latency.

All numbers are computed directly from the released AlzKG and the AlzBench
probe queries; nothing is fabricated.
"""

import argparse
import time

from alzgraph.common import read_json, write_json
from alzgraph.metrics import _tokens
from alzgraph.retrieval import AlzGraphRetriever


def load_probes() -> list[dict]:
    probes = []
    for item in read_json("data/alzbench/t1/mcq.json"):
        probes.append({"query": item["question"], "gold": item.get("answer", "")})
    for item in read_json("data/alzbench/t3/bpm_mcq.json"):
        probes.append({"query": item["clinical_scenario"], "gold": item.get("recommended", "")})
    return probes


def covered(gold: str, paths: list[str]) -> bool:
    gold_tokens = [t for t in _tokens(gold) if len(t) > 2]
    if not gold_tokens:
        return False
    evidence = set(_tokens(" ".join(paths)))
    hits = sum(1 for t in gold_tokens if t in evidence)
    return hits / len(gold_tokens) >= 0.5


def run(probes: list[dict], nodes: int, depth: int) -> dict:
    r = AlzGraphRetriever("data/alzkg/triplets.json", max_subgraph_nodes=nodes, max_paths=12, max_depth=depth)
    n_paths, n_cov, latencies = 0, 0, []
    for p in probes:
        t0 = time.perf_counter()
        out = r.retrieve(p["query"])
        latencies.append((time.perf_counter() - t0) * 1000)
        n_paths += len(out["paths"])
        n_cov += int(covered(p["gold"], out["paths"]))
    n = max(len(probes), 1)
    return {
        "max_subgraph_nodes": nodes,
        "max_depth": depth,
        "avg_paths": round(n_paths / n, 2),
        "gold_coverage": round(n_cov / n, 3),
        "mean_latency_ms": round(sum(latencies) / n, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-RAG retrieval ablation over AlzKG.")
    parser.add_argument("--out", default="data/alzkg/retrieval_ablation.json")
    args = parser.parse_args()
    probes = load_probes()
    results = {"n_probes": len(probes), "subgraph_sweep": [], "depth_sweep": []}
    print(f"Probes: {len(probes)}")
    print("\n[subgraph-size sweep @ depth=4]")
    print(f"{'nodes':>6} {'avg_paths':>10} {'gold_cov':>9} {'latency_ms':>11}")
    for nodes in [10, 20, 30, 40, 50]:
        row = run(probes, nodes, 4)
        results["subgraph_sweep"].append(row)
        print(f"{nodes:>6} {row['avg_paths']:>10} {row['gold_coverage']:>9} {row['mean_latency_ms']:>11}")
    print("\n[path-depth sweep @ nodes=30]")
    print(f"{'depth':>6} {'avg_paths':>10} {'gold_cov':>9} {'latency_ms':>11}")
    for depth in [1, 2, 3, 4]:
        row = run(probes, 30, depth)
        results["depth_sweep"].append(row)
        print(f"{depth:>6} {row['avg_paths']:>10} {row['gold_coverage']:>9} {row['mean_latency_ms']:>11}")
    write_json(results, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
