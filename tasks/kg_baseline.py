"""AlzBench KG-only baseline (no LLM, fully reproducible).

This reference baseline answers AlzBench multiple-choice items using *only*
AlzKG evidence -- no language model and no network access. For each item it:

  1. recognizes the AD entities mentioned in the question stem (lexicon NER);
  2. runs a literature-weighted personalized PageRank (PPR) over AlzKG seeded
     from those question entities -- the same retriever used by the Graph-RAG
     setting; and
  3. scores every option by the PPR evidence mass flowing to the AlzKG entities
     named in that option, breaking ties with the direct edge evidence
     (co-mention paper counts) between question and option entities.

The highest-scoring option is selected. Because the procedure is deterministic
and depends on nothing but the released graph, anyone can reproduce the exact
numbers it reports. It is a *graph-grounding* baseline -- a measured floor for
"how far does AlzKG evidence alone get you" -- not a substitute for an LLM.

Usage:
    python tasks/kg_baseline.py --dataset data/alzbench/t1/mcq.json
    python tasks/kg_baseline.py --dataset data/alzbench/t3/bpm_mcq.json \
        --question-field clinical_scenario --out runs/t3_kg_baseline.json
"""

import argparse
import re
from pathlib import Path

from alzgraph.common import option_letter, read_json, write_json
from alzgraph.lexicon import detect
from alzgraph.retrieval import AlzGraphRetriever

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]


def _question_text(item: dict, field: str | None) -> str:
    if field and item.get(field):
        return item[field]
    for key in ("question", "clinical_scenario", "scenario", "prompt"):
        if item.get(key):
            return item[key]
    return ""


def _option_text(option: str) -> str:
    """Strip a leading 'A) ' / 'B. ' style label from an option string."""
    return re.sub(r"^\s*[A-Fa-f][\).\:]\s*", "", option).strip()


def _entities_in(text: str) -> set[str]:
    """Lowercased AlzKG node ids recognized in ``text`` via the lexicon."""
    found: set[str] = set()
    for canon_set in detect(text).values():
        for canon in canon_set:
            found.add(canon.lower())
    return found


class KGBaseline:
    def __init__(self, triplets_path: str) -> None:
        self.retriever = AlzGraphRetriever(triplets_path)
        self.nodes = self.retriever.nodes
        # direct edge evidence (paper_count) between any ordered pair, both ways
        self.edge_weight: dict[tuple[str, str], float] = {}
        for head, edges in self.retriever.out_edges.items():
            for e in edges:
                pc = float(e.get("paper_count", 1) or 1)
                pair = (head, e["tail"])
                self.edge_weight[pair] = max(self.edge_weight.get(pair, 0.0), pc)

    def _direct_evidence(self, seeds: set[str], targets: set[str]) -> float:
        total = 0.0
        for s in seeds:
            for t in targets:
                total += self.edge_weight.get((s, t), 0.0)
                total += self.edge_weight.get((t, s), 0.0)
        return total

    def answer(self, item: dict, question_field: str | None) -> dict:
        question = _question_text(item, question_field)
        options = item.get("options", [])
        seeds = {e for e in _entities_in(question) if e in self.nodes}
        ppr = self.retriever._pagerank(seeds) if seeds else {}

        scored = []
        for idx, option in enumerate(options):
            targets = {e for e in _entities_in(_option_text(option)) if e in self.nodes}
            ppr_mass = sum(ppr.get(t, 0.0) for t in targets)
            direct = self._direct_evidence(seeds, targets)
            scored.append(
                {
                    "letter": OPTION_LETTERS[idx] if idx < len(OPTION_LETTERS) else str(idx),
                    "text": option,
                    "entities": sorted(targets),
                    "ppr_mass": ppr_mass,
                    "direct_evidence": direct,
                }
            )

        # Rank by PPR evidence mass, then direct co-mention evidence, then order.
        ranked = sorted(
            scored,
            key=lambda s: (s["ppr_mass"], s["direct_evidence"], -OPTION_LETTERS.index(s["letter"])),
            reverse=True,
        )
        best = ranked[0] if ranked else None
        grounded = bool(best and (best["ppr_mass"] > 0 or best["direct_evidence"] > 0))
        return {
            "pred_option": best["letter"] if best else None,
            "grounded": grounded,
            "seeds": sorted(seeds),
            "option_scores": scored,
        }


def evaluate(args: argparse.Namespace) -> None:
    data = read_json(args.dataset)
    if args.sample:
        data = data[: args.sample]
    baseline = KGBaseline(args.triplets)

    rows = []
    n_correct = 0
    n_grounded = 0
    n_grounded_correct = 0
    for item in data:
        res = baseline.answer(item, args.question_field)
        gold = item.get("correct_answer") or option_letter(item.get("answer", ""))
        correct = float(res["pred_option"] == gold) if gold else None
        if correct == 1.0:
            n_correct += 1
        if res["grounded"]:
            n_grounded += 1
            if correct == 1.0:
                n_grounded_correct += 1
        rows.append(
            {
                "id": item.get("id"),
                "pred_option": res["pred_option"],
                "gold_option": gold,
                "correct": correct,
                "grounded": res["grounded"],
                "seeds": res["seeds"],
                "option_scores": res["option_scores"],
            }
        )

    n = len(rows)
    summary = {
        "dataset": args.dataset,
        "method": "kg_only_baseline",
        "n_items": n,
        "accuracy": round(n_correct / n, 4) if n else 0.0,
        "coverage": round(n_grounded / n, 4) if n else 0.0,
        "accuracy_on_grounded": round(n_grounded_correct / n_grounded, 4) if n_grounded else 0.0,
        "random_baseline": round(
            sum(1 / len(it.get("options", [1])) for it in data) / n, 4
        )
        if n
        else 0.0,
    }
    out = {"summary": summary, "predictions": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(out, args.out)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlzBench KG-only MCQ baseline (no LLM).")
    parser.add_argument("--dataset", required=True, help="AlzBench MCQ JSON.")
    parser.add_argument("--triplets", default="data/alzkg/triplets.json")
    parser.add_argument(
        "--question-field",
        default=None,
        help="Field holding the question stem (default: auto-detect question/clinical_scenario).",
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--out", default="runs/kg_baseline_predictions.json")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
