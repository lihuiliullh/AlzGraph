"""Mine AlzKG from the real PubMed corpus (data/corpus/abstracts.jsonl).

Pipeline (mirrors the EpiGraph rule-based co-occurrence stage, on real data):
  1. recognize AlzKG entities in each abstract (title + abstract + MeSH) using the
     dictionary NER lexicon;
  2. for every pair of co-occurring entities from two different layers, create a
     typed, directed relation (oriented by layer pair);
  3. paper_count = number of distinct abstracts (PMIDs) supporting the pair --- a
     TRUE literature count;
  4. keep relations supported by at least ``--min_papers`` abstracts.

Outputs data/alzkg/triplets.json, kg_stats.json, entity_frequency.json, and the
project-page demo graph docs/data/demo_graph.json.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from alzgraph.common import stable_id, write_json
from alzgraph.lexicon import LEXICON, detect

LAYER_ORDER = ["gene", "biomarker", "stage", "treatment", "outcome"]
LAYER_COLOR = {"gene": "#7C3AED", "biomarker": "#0EA5E9", "stage": "#EAB308", "treatment": "#14B8A6", "outcome": "#EF4444"}
LAYER_SOURCE = {"gene": "OMIM/GWAS", "biomarker": "NIA-AA/MeSH", "stage": "NIA-AA/UMLS", "treatment": "ChEBI/AAN", "outcome": "HPO/MeSH"}

# One canonical orientation + relation for each of the 10 cross-layer pairs.
ORIENT = {
    frozenset({"gene", "biomarker"}): ("gene", "biomarker", "modulates_biomarker"),
    frozenset({"gene", "stage"}): ("gene", "stage", "risk_gene_for"),
    frozenset({"gene", "treatment"}): ("gene", "treatment", "pharmacogenomic_consideration"),
    frozenset({"gene", "outcome"}): ("gene", "outcome", "gene_associated_outcome"),
    frozenset({"biomarker", "stage"}): ("stage", "biomarker", "characterized_by_biomarker"),
    frozenset({"biomarker", "treatment"}): ("biomarker", "treatment", "biomarker_guides_treatment"),
    frozenset({"biomarker", "outcome"}): ("biomarker", "outcome", "predicts_outcome"),
    frozenset({"stage", "treatment"}): ("stage", "treatment", "treated_with"),
    frozenset({"stage", "outcome"}): ("stage", "outcome", "associated_outcome"),
    frozenset({"treatment", "outcome"}): ("treatment", "outcome", "has_outcome"),
}


def mine(corpus_path: str, min_papers: int):
    edges_pmids: dict = defaultdict(set)          # (h,r,t,hl,tl) -> {pmid}
    entity_df: Counter = Counter()                 # entity -> #abstracts
    layer_of = {c: s["layer"] for c, s in LEXICON.items()}
    n_docs = 0
    for line in Path(corpus_path).open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        n_docs += 1
        text = f"{rec.get('title','')}. {rec.get('abstract','')} {' '.join(rec.get('mesh', []))}"
        found = detect(text)
        present = sorted({(e, layer) for layer, ents in found.items() for e in ents})
        for e, _ in present:
            entity_df[e] += 1
        pmid = rec.get("pmid", "")
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                (ea, la), (eb, lb) = present[i], present[j]
                if la == lb:
                    continue
                hl, tl, rel = ORIENT[frozenset({la, lb})]
                head, tail = (ea, eb) if la == hl else (eb, ea)
                edges_pmids[(head, rel, tail, hl, tl)].add(pmid)

    rows = []
    for (head, rel, tail, hl, tl), pmids in edges_pmids.items():
        if len(pmids) < min_papers:
            continue
        rows.append({
            "id": stable_id(head, rel, tail, prefix="kg"),
            "head": head, "relation": rel, "tail": tail,
            "head_layer": hl, "tail_layer": tl,
            "paper_count": len(pmids), "weight_label": "papers",
            "head_source": LEXICON.get(head, {}).get("source", LAYER_SOURCE[hl]),
            "tail_source": LEXICON.get(tail, {}).get("source", LAYER_SOURCE[tl]),
            "paper_ids": sorted(pmids)[:25],
        })
    rows.sort(key=lambda r: (-r["paper_count"], r["head"], r["tail"]))
    return rows, entity_df, n_docs


def compute_stats(triplets, entity_df, n_docs, min_papers):
    nodes, degree, rel_counts, pair_counts = set(), Counter(), Counter(), Counter()
    for t in triplets:
        nodes.add(t["head"]); nodes.add(t["tail"])
        degree[t["head"]] += 1; degree[t["tail"]] += 1
        rel_counts[t["relation"]] += 1
        pair_counts[f"{t['head_layer']}->{t['tail_layer']}"] += 1
    pcs = [t["paper_count"] for t in triplets]
    per_layer = Counter(LEXICON[n]["layer"] for n in nodes if n in LEXICON)
    return {
        "corpus_abstracts": n_docs,
        "min_papers_threshold": min_papers,
        "n_entities": len(nodes),
        "n_triplets": len(triplets),
        "n_cross_layer_triplets": len(triplets),
        "n_relation_types": len(rel_counts),
        "n_layers": len(LAYER_ORDER),
        "entities_per_layer": {l: per_layer.get(l, 0) for l in LAYER_ORDER},
        "relation_type_counts": dict(rel_counts.most_common()),
        "layer_pair_counts": dict(pair_counts.most_common()),
        "paper_count_min": min(pcs) if pcs else 0,
        "paper_count_median": statistics.median(pcs) if pcs else 0,
        "paper_count_max": max(pcs) if pcs else 0,
        "paper_count_mean": round(statistics.mean(pcs), 1) if pcs else 0,
        "top_degree_entities": dict(degree.most_common(15)),
        "top_mentioned_entities": dict(entity_df.most_common(15)),
    }


def build_demo_graph(triplets, max_nodes=60, max_links=150):
    degree = Counter()
    for t in triplets:
        degree[t["head"]] += t["paper_count"]
        degree[t["tail"]] += t["paper_count"]
    keep = {n for n, _ in degree.most_common(max_nodes)}
    sub = [t for t in triplets if t["head"] in keep and t["tail"] in keep]
    sub.sort(key=lambda r: -r["paper_count"])
    sub = sub[:max_links]
    used = {t["head"] for t in sub} | {t["tail"] for t in sub}
    deg = Counter()
    for t in sub:
        deg[t["head"]] += 1; deg[t["tail"]] += 1
    nodes = [{
        "id": n, "label": n, "layer": LEXICON[n]["layer"],
        "source": LAYER_SOURCE[LEXICON[n]["layer"]],
        "color": LAYER_COLOR[LEXICON[n]["layer"]], "degree": deg[n],
    } for n in sorted(used, key=lambda x: -deg[x])]
    links = [{"source": t["head"], "target": t["tail"], "relation": t["relation"],
              "papers": t["paper_count"], "evidence": f"{t['paper_count']} co-mentioning abstracts"} for t in sub]
    return {"meta": {"name": "AlzKG (literature-mined) subgraph",
                     "description": "Top-degree subgraph of AlzKG, mined by cross-layer co-occurrence over PubMed abstracts.",
                     "nodes": len(nodes), "links": len(links), "layers": LAYER_ORDER, "layer_color": LAYER_COLOR},
            "nodes": nodes, "links": links}


def main():
    ap = argparse.ArgumentParser(description="Mine AlzKG from the PubMed corpus.")
    ap.add_argument("--corpus", default="data/corpus/abstracts.jsonl")
    ap.add_argument("--out_dir", default="data/alzkg")
    ap.add_argument("--docs_dir", default="docs/data")
    ap.add_argument("--min_papers", type=int, default=5)
    args = ap.parse_args()

    triplets, entity_df, n_docs = mine(args.corpus, args.min_papers)
    stats = compute_stats(triplets, entity_df, n_docs, args.min_papers)
    write_json(triplets, Path(args.out_dir) / "triplets.json")
    write_json(stats, Path(args.out_dir) / "kg_stats.json")
    write_json(dict(entity_df.most_common()), Path(args.out_dir) / "entity_frequency.json")
    write_json(build_demo_graph(triplets), Path(args.docs_dir) / "demo_graph.json")

    print(f"Literature-mined AlzKG built from {n_docs} abstracts (min_papers={args.min_papers}).")
    print(f"  entities : {stats['n_entities']}")
    print(f"  triplets : {stats['n_triplets']} (all cross-layer)")
    print(f"  relation types : {stats['n_relation_types']}")
    print(f"  paper_count (min/median/max): {stats['paper_count_min']}/{stats['paper_count_median']}/{stats['paper_count_max']}")
    print(f"  entities/layer : {stats['entities_per_layer']}")
    print(f"  top degree : {list(stats['top_degree_entities'].items())[:6]}")


if __name__ == "__main__":
    main()
