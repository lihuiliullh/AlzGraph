"""Materialize the released seed AlzKG from the curated ontology.

Outputs (all statistics are computed from the curated edge table; nothing is
fabricated):
  - data/alzkg/triplets.json        the released seed knowledge graph
  - data/alzkg/kg_stats.json        computed graph statistics (used by the paper)
  - data/alzkg/paper_metadata.json  source-provenance manifest
  - docs/data/demo_graph.json       compact graph for the interactive project page

For curated seed edges, ``paper_count`` carries the curated evidence tier
(1 = emerging, 2 = established, 3 = guideline/landmark) and ``weight_label`` is
"tier". The PMC builder in ``alzgraph/build_kg.py`` instead populates true
literature paper counts.
"""

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from alzgraph.common import stable_id, write_json
from alzgraph.ontology import CURATED_EDGES, LAYERS, SOURCES, layer_of

LAYER_ORDER = ["gene", "biomarker", "stage", "treatment", "outcome"]
LAYER_COLOR = {
    "gene": "#7C3AED",
    "biomarker": "#0EA5E9",
    "stage": "#EAB308",
    "treatment": "#14B8A6",
    "outcome": "#EF4444",
}


def build_triplets() -> list[dict]:
    rows = []
    for head, relation, tail, tier, note in CURATED_EDGES:
        hl, tl = layer_of(head), layer_of(tail)
        if hl == "unknown" or tl == "unknown":
            raise ValueError(f"Edge uses entity outside the layer vocabulary: {head} -> {tail}")
        rows.append(
            {
                "id": stable_id(head, relation, tail, prefix="kg"),
                "head": head,
                "relation": relation,
                "tail": tail,
                "head_layer": hl,
                "tail_layer": tl,
                "paper_count": tier,
                "weight_label": "tier",
                "evidence_tier": tier,
                "evidence": note,
                "head_source": SOURCES.get(head, "curator"),
                "tail_source": SOURCES.get(tail, "curator"),
            }
        )
    return sorted(rows, key=lambda r: (-r["evidence_tier"], r["head"], r["tail"]))


def compute_stats(triplets: list[dict]) -> dict:
    nodes = set()
    degree: Counter = Counter()
    relation_counts: Counter = Counter()
    layer_pair_counts: Counter = Counter()
    cross_layer = 0
    for t in triplets:
        nodes.add(t["head"])
        nodes.add(t["tail"])
        degree[t["head"]] += 1
        degree[t["tail"]] += 1
        relation_counts[t["relation"]] += 1
        layer_pair_counts[f"{t['head_layer']}->{t['tail_layer']}"] += 1
        if t["head_layer"] != t["tail_layer"]:
            cross_layer += 1
    entities_per_layer = {layer: len(members) for layer, members in LAYERS.items()}
    tiers = [t["evidence_tier"] for t in triplets]
    return {
        "n_entities": len(nodes),
        "n_entities_defined": sum(entities_per_layer.values()),
        "n_triplets": len(triplets),
        "n_cross_layer_triplets": cross_layer,
        "cross_layer_fraction": round(cross_layer / max(len(triplets), 1), 3),
        "n_relation_types": len(relation_counts),
        "n_layers": len(LAYERS),
        "entities_per_layer": entities_per_layer,
        "relation_type_counts": dict(relation_counts.most_common()),
        "layer_pair_counts": dict(layer_pair_counts.most_common()),
        "median_evidence_tier": statistics.median(tiers) if tiers else 0,
        "tier_distribution": dict(Counter(tiers)),
        "n_sources": len(set(SOURCES.values())),
        "sources": sorted(set(SOURCES.values())),
        "top_degree_entities": dict(degree.most_common(12)),
    }


def build_demo_graph(triplets: list[dict]) -> dict:
    degree: Counter = Counter()
    for t in triplets:
        degree[t["head"]] += 1
        degree[t["tail"]] += 1
    node_ids = {t["head"] for t in triplets} | {t["tail"] for t in triplets}
    nodes = [
        {
            "id": n,
            "label": n,
            "layer": layer_of(n),
            "source": SOURCES.get(n, "curator"),
            "color": LAYER_COLOR.get(layer_of(n), "#64748B"),
            "degree": degree[n],
        }
        for n in sorted(node_ids, key=lambda x: -degree[x])
    ]
    links = [
        {
            "source": t["head"],
            "target": t["tail"],
            "relation": t["relation"],
            "tier": t["evidence_tier"],
            "evidence": t["evidence"],
        }
        for t in triplets
    ]
    return {
        "meta": {
            "name": "AlzGraph seed subgraph",
            "description": "The released AlzKG seed graph, curated from public ontologies and AD clinical guidelines.",
            "nodes": len(nodes),
            "links": len(links),
            "layers": LAYER_ORDER,
            "layer_color": LAYER_COLOR,
        },
        "nodes": nodes,
        "links": links,
    }


def build_provenance() -> dict:
    by_source = defaultdict(list)
    for entity, src in SOURCES.items():
        by_source[src].append(entity)
    return {
        "description": "Source-provenance manifest for the curated seed AlzKG.",
        "resources": {
            "OMIM": "Gene-disease associations for causal/risk genes",
            "ADSP": "Alzheimer's Disease Sequencing Project rare-variant genes",
            "GWAS_Catalog": "Common-variant late-onset AD risk loci",
            "NIA_AA": "NIA-AA ATN biomarker framework and staging",
            "MeSH": "Imaging, cognitive instruments, and clinical terms",
            "HPO": "Phenotype and adverse-event vocabulary",
            "ChEBI": "Small-molecule drug identifiers",
            "AAN_AA_AUC": "AAN / Alzheimer's Association appropriate-use criteria for anti-amyloid mAbs",
            "AAN_guideline": "AAN dementia management guidelines",
            "UMLS": "Cross-ontology linking for atypical AD syndromes",
        },
        "entities_by_source": {src: sorted(ents) for src, ents in sorted(by_source.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the released seed AlzKG from the curated ontology.")
    parser.add_argument("--out_dir", default="data/alzkg")
    parser.add_argument("--docs_dir", default="docs/data")
    args = parser.parse_args()

    triplets = build_triplets()
    stats = compute_stats(triplets)
    write_json(triplets, Path(args.out_dir) / "triplets.json")
    write_json(stats, Path(args.out_dir) / "kg_stats.json")
    write_json(build_provenance(), Path(args.out_dir) / "paper_metadata.json")
    write_json(build_demo_graph(triplets), Path(args.docs_dir) / "demo_graph.json")

    print("Released seed AlzKG built.")
    print(f"  entities            : {stats['n_entities']}")
    print(f"  triplets            : {stats['n_triplets']}")
    print(f"  cross-layer triplets: {stats['n_cross_layer_triplets']} ({stats['cross_layer_fraction']*100:.1f}%)")
    print(f"  relation types      : {stats['n_relation_types']}")
    print(f"  sources             : {stats['n_sources']}")
    print(f"  entities/layer      : {stats['entities_per_layer']}")


if __name__ == "__main__":
    main()
