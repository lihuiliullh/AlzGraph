"""Lightweight AlzKG preview builder from local PMC/PubMed XML files.

This builder mirrors the full-scale construction pipeline at small scale: it
detects AlzKG entities by layer in each article, links co-occurring cross-layer
entity pairs using the ontology relation hints, and counts the number of papers
supporting each triplet (true literature ``paper_count``).

For the curated, guideline-grounded seed graph used by the project page and the
benchmark tasks, see ``scripts/build_seed_kg.py``.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .common import stable_id, write_json
from .ontology import LAYERS, RELATION_HINTS, SOURCES


def parse_pmc_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    text = " ".join(root.itertext())
    title = (
        " ".join(root.findall(".//article-title")[0].itertext())
        if root.findall(".//article-title")
        else path.stem
    )
    return {"paper_id": path.stem, "title": re.sub(r"\s+", " ", title), "text": re.sub(r"\s+", " ", text)}


def detect_entities(text: str) -> dict:
    lower = text.lower()
    out = {}
    for layer, terms in LAYERS.items():
        hits = [term for term in terms if term.lower() in lower]
        out[layer] = sorted(set(hits))
    return out


def build_triplets(papers: list[dict]) -> list[dict]:
    evidence: dict = {}
    for paper in papers:
        entities = detect_entities(paper["text"])
        for (src_layer, dst_layer), relation in RELATION_HINTS.items():
            for head in entities[src_layer]:
                for tail in entities[dst_layer]:
                    if head.lower() == tail.lower():
                        continue
                    key = (head, relation, tail, src_layer, dst_layer)
                    evidence.setdefault(key, set()).add(paper["paper_id"])
    rows = []
    for (head, relation, tail, head_layer, tail_layer), paper_ids in evidence.items():
        rows.append(
            {
                "id": stable_id(head, relation, tail, prefix="kg"),
                "head": head,
                "relation": relation,
                "tail": tail,
                "head_layer": head_layer,
                "tail_layer": tail_layer,
                "paper_count": len(paper_ids),
                "weight_label": "papers",
                "head_source": SOURCES.get(head, "literature"),
                "tail_source": SOURCES.get(tail, "literature"),
                "paper_ids": sorted(paper_ids),
            }
        )
    return sorted(rows, key=lambda x: (-x["paper_count"], x["head"], x["tail"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight AlzKG preview from PMC XML files.")
    parser.add_argument("--pmc_dir", required=True, help="Directory containing PMC XML files.")
    parser.add_argument("--out_dir", default="data/alzkg", help="Output directory.")
    args = parser.parse_args()

    pmc_dir = Path(args.pmc_dir)
    papers = [parse_pmc_xml(path) for path in sorted(pmc_dir.glob("*.xml"))]
    triplets = build_triplets(papers)
    metadata = [
        {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "entity_counts": Counter({k: len(v) for k, v in detect_entities(paper["text"]).items()}),
        }
        for paper in papers
    ]
    out_dir = Path(args.out_dir)
    write_json(triplets, out_dir / "triplets.json")
    write_json(metadata, out_dir / "paper_metadata.json")
    print(json.dumps({"papers": len(papers), "triplets": len(triplets)}, indent=2))


if __name__ == "__main__":
    main()
