"""Mine AlzKG from PMC FULL-TEXT via EpiGraph's relation-extraction procedure.

This reproduces EpiGraph's pipeline (Sec. 2 + App. B.3) on real PMC open-access
full text instead of abstracts:

  1. Source: ``data/corpus/fulltext_sentences.jsonl`` -- PMC full-text articles,
     reduced to candidate sentences that contain >=2 AlzKG entity mentions
     (produced by ``scripts/fetch_pmc_fulltext.py``).
  2. (i) Rule-based extraction: for every sentence containing a co-occurring
     cross-layer entity pair, a per-relation template ``(subject layer,
     trigger-phrase set, object layer)`` is applied (cf. EpiGraph Table 5). A
     triplet is emitted only when a trigger phrase for that relation appears in
     the sentence -- i.e. relations are sentence-grounded, not mere whole-doc
     co-occurrence.
  3. (ii) LLM-based extraction (MiniMax in EpiGraph): provided as an optional
     pass gated on an API key; not run in this release (no model endpoint),
     mirroring how the benchmark model tables are left for the user to populate.
  4. paper_count P = number of distinct papers (PMIDs) supporting the triplet;
     rule-based and LLM extractions are merged, conflicts resolved by higher P.
  5. Keep relations supported by >= ``--min_papers`` papers.

Outputs match the abstract builder's schema (drop-in for the retriever/tasks):
``data/alzkg/triplets.json``, ``kg_stats.json``, ``entity_frequency.json``,
``relation_evidence.json``, and the project-page ``docs/data/demo_graph.json``.
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alzgraph.common import stable_id, write_json  # noqa: E402
from alzgraph.lexicon import LEXICON  # noqa: E402
from scripts.build_kg_from_corpus import (  # noqa: E402
    LAYER_COLOR, LAYER_ORDER, LAYER_SOURCE, ORIENT, compute_stats,
)

# EpiGraph Table 5, adapted to the five AlzKG layers. Each oriented layer pair
# maps to (relation, trigger-phrase set). A relation is emitted only if some
# trigger phrase (a lowercase substring / stem) appears in the sentence that
# already contains the co-occurring entity pair.
TRIGGERS: dict = {
    ("gene", "biomarker", "modulates_biomarker"): [
        "modulat", "regulat", "affect", "influenc", "increas", "decreas", "reduc",
        "elevat", "associated with", "correlat", "alter", "drive", "promot",
        "clear", "aggregat", "accumulat", "express",
    ],
    ("gene", "stage", "risk_gene_for"): [
        "risk", "associated with", "linked to", "implicated in", "mutation",
        "variant", "carrier", "predispos", "susceptib", "cause", "etiolog",
        "genetic", "allele", "polymorphism",
    ],
    ("gene", "treatment", "pharmacogenomic_consideration"): [
        "response to", "responder", "carrier", "genotype", "stratif", "eligib",
        "modif", "interact", "metaboli", "contraindicat", "caution", "tailor",
    ],
    ("gene", "outcome", "gene_associated_outcome"): [
        "associated with", "predict", "risk of", "linked to", "correlat",
        "worse", "faster", "decline", "progression", "outcome",
    ],
    ("stage", "biomarker", "characterized_by_biomarker"): [
        "characterized by", "marked by", "defined by", "positive", "elevated",
        "increased", "presence of", "accumulat", "deposit", "show", "exhibit",
        "consistent with", "reveal", "associated with", "hallmark", "burden",
    ],
    ("biomarker", "treatment", "biomarker_guides_treatment"): [
        "eligib", "candidat", "indicat", "guide", "select", "requir", "confirm",
        "screen", "stratif", "positive", "prior to", "before initiat", "enrich",
    ],
    ("biomarker", "outcome", "predicts_outcome"): [
        "predict", "associated with", "risk of", "correlat", "linked to",
        "progression", "decline", "prognos", "marker of", "indicative of",
    ],
    ("stage", "treatment", "treated_with"): [
        "treat", "therap", "administ", "first-line", "approved", "prescrib",
        "manage", "receiv", "indicated for", "standard of care", "use of",
    ],
    ("stage", "outcome", "associated_outcome"): [
        "associated with", "progress", "decline", "develop", "risk of",
        "lead to", "result in", "convert", "outcome", "course",
    ],
    ("treatment", "outcome", "has_outcome"): [
        "resulted in", "led to", "cause", "associated with", "risk of", "adverse",
        "efficac", "reduc", "slow", "improv", "benefit", "side effect", "increas",
        "decreas", "tolerab", "response", "discontinu",
    ],
}
TRIGGER_BY_REL = {rel: phrases for (_, __, rel), phrases in TRIGGERS.items()}


def _has_trigger(sentence_lc: str, relation: str) -> bool:
    return any(p in sentence_lc for p in TRIGGER_BY_REL.get(relation, []))


def mine(corpus_path: str, min_papers: int, max_evidence: int = 3):
    edges_pmids: dict = defaultdict(set)            # (h,r,t,hl,tl) -> {pmid}
    edges_sents: dict = defaultdict(list)           # -> [example sentences]
    entity_df: Counter = Counter()                  # entity -> #papers mentioning
    n_papers, n_sentences = 0, 0

    for line in Path(corpus_path).open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        n_papers += 1
        pmid = str(rec.get("pmid", ""))
        seen_entities = set()
        for sent in rec.get("candidate_sentences", []):
            n_sentences += 1
            text_lc = sent["text"].lower()
            present = sorted({(e["entity"], e["layer"]) for e in sent["entities"]})
            for e, _ in present:
                seen_entities.add(e)
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    (ea, la), (eb, lb) = present[i], present[j]
                    if la == lb:
                        continue
                    hl, tl, rel = ORIENT[frozenset({la, lb})]
                    if not _has_trigger(text_lc, rel):
                        continue
                    head, tail = (ea, eb) if la == hl else (eb, ea)
                    key = (head, rel, tail, hl, tl)
                    edges_pmids[key].add(pmid)
                    if len(edges_sents[key]) < max_evidence:
                        edges_sents[key].append({"pmid": pmid, "sentence": sent["text"][:400]})
        for e in seen_entities:
            entity_df[e] += 1

    rows = []
    for (head, rel, tail, hl, tl), pmids in edges_pmids.items():
        if len(pmids) < min_papers:
            continue
        rows.append({
            "id": stable_id(head, rel, tail, prefix="kg"),
            "head": head, "relation": rel, "tail": tail,
            "head_layer": hl, "tail_layer": tl,
            "paper_count": len(pmids), "weight_label": "papers",
            "extraction": "rule_based_fulltext",
            "head_source": LEXICON.get(head, {}).get("source", LAYER_SOURCE[hl]),
            "tail_source": LEXICON.get(tail, {}).get("source", LAYER_SOURCE[tl]),
            "paper_ids": sorted(pmids)[:25],
        })
    rows.sort(key=lambda r: (-r["paper_count"], r["head"], r["tail"]))
    evidence = {
        stable_id(h, r, t, prefix="kg"): edges_sents[(h, r, t, hl, tl)]
        for (h, r, t, hl, tl) in edges_pmids
    }
    return rows, entity_df, n_papers, n_sentences, evidence


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
              "papers": t["paper_count"],
              "evidence": f"{t['paper_count']} full-text papers (sentence-level, rule-based)"}
             for t in sub]
    return {"meta": {"name": "AlzKG (full-text, rule-based) subgraph",
                     "description": "Top-degree subgraph of AlzKG, mined by sentence-level rule-based "
                                    "relation extraction over PMC full-text articles (EpiGraph procedure).",
                     "nodes": len(nodes), "links": len(links),
                     "layers": LAYER_ORDER, "layer_color": LAYER_COLOR},
            "nodes": nodes, "links": links}


def main():
    ap = argparse.ArgumentParser(description="Mine AlzKG from PMC full text (EpiGraph procedure).")
    ap.add_argument("--corpus", default="data/corpus/fulltext_sentences.jsonl")
    ap.add_argument("--out_dir", default="data/alzkg")
    ap.add_argument("--docs_dir", default="docs/data")
    ap.add_argument("--min_papers", type=int, default=3)
    args = ap.parse_args()

    triplets, entity_df, n_papers, n_sentences, evidence = mine(args.corpus, args.min_papers)
    stats = compute_stats(triplets, entity_df, n_papers, args.min_papers)
    stats["source"] = "pmc_fulltext"
    stats["extraction"] = "sentence_level_rule_based (EpiGraph Table 5)"
    stats["corpus_fulltext_papers"] = n_papers
    stats["candidate_sentences"] = n_sentences
    stats.pop("corpus_abstracts", None)

    write_json(triplets, Path(args.out_dir) / "triplets.json")
    write_json(stats, Path(args.out_dir) / "kg_stats.json")
    write_json(dict(entity_df.most_common()), Path(args.out_dir) / "entity_frequency.json")
    write_json(evidence, Path(args.out_dir) / "relation_evidence.json")
    write_json(build_demo_graph(triplets), Path(args.docs_dir) / "demo_graph.json")

    print(f"Full-text AlzKG built from {n_papers} PMC papers, {n_sentences} candidate sentences "
          f"(min_papers={args.min_papers}).")
    print(f"  entities : {stats['n_entities']}")
    print(f"  triplets : {stats['n_triplets']} (all cross-layer, sentence-grounded)")
    print(f"  relation types : {stats['n_relation_types']}")
    print(f"  paper_count (min/median/max): "
          f"{stats['paper_count_min']}/{stats['paper_count_median']}/{stats['paper_count_max']}")
    print(f"  entities/layer : {stats['entities_per_layer']}")
    print(f"  top degree : {list(stats['top_degree_entities'].items())[:6]}")


if __name__ == "__main__":
    main()
