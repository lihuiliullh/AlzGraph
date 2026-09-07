"""Sample random AlzKG triplets and resolve each one's grounding sentence,
for a manual precision audit (see paper Sec. 2.5).

This script performs only the deterministic, mechanical half of the audit:
drawing a reproducible random sample of triplets and locating, for each,
the actual corpus sentence whose recognized-entity list contains both the
head and tail entity (i.e. the sentence the extractor used to emit that
edge). It does not judge correctness -- that half is inherently manual and
is reported in data/alzkg/extraction_audit.json and paper Sec. 2.5.

Usage:
    python scripts/audit_kg_sample.py --n 30 --seed 42 --out /tmp/audit_sample.json
"""

import argparse
import json
import random

from alzgraph.common import read_json, write_json


def resolve_grounding_sentence(triplet: dict, sentences_path: str) -> str | None:
    """Find the first candidate sentence (from the triplet's first supporting
    paper) whose recognized entities include both the head and the tail."""
    pmid = triplet["paper_ids"][0]
    head, tail = triplet["head"].lower(), triplet["tail"].lower()
    with open(sentences_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["pmid"] != pmid:
                continue
            for s in rec["candidate_sentences"]:
                names = {e["entity"].lower() for e in s.get("entities", [])}
                if head in names and tail in names:
                    return s["text"]
            break
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample AlzKG triplets for a manual precision audit.")
    parser.add_argument("--triplets", default="data/alzkg/triplets.json")
    parser.add_argument("--sentences", default="data/corpus/fulltext_sentences.jsonl")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/alzkg/extraction_audit_sample.json")
    args = parser.parse_args()

    triplets = read_json(args.triplets)
    random.seed(args.seed)
    sample = random.sample(triplets, args.n)

    rows = []
    for t in sample:
        rows.append(
            {
                "triplet": f"{t['head']} --[{t['relation']}]--> {t['tail']}",
                "paper_count": t["paper_count"],
                "pmid_checked": t["paper_ids"][0],
                "grounding_sentence": resolve_grounding_sentence(t, args.sentences),
            }
        )
    write_json(rows, args.out)
    print(f"Sampled {len(rows)} triplets -> {args.out}. Manual correctness judgment is a separate step.")


if __name__ == "__main__":
    main()
