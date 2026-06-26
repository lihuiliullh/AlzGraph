"""AlzBench Task 2: Clinical Report Generation.

Generate a neurologist-style diagnostic impression from a patient's cognitive
assessment and fluid/imaging biomarker panel. ADNI and memory-clinic notes are
access-restricted, so this release ships a local JSONL adapter that preserves the
schema, build logic, and evaluation interface without redistributing patient data.

Prepare a local JSONL with fields:
  patient_history, cognitive_assessment, biomarker_panel, impression
"""

import argparse
import json
from pathlib import Path

from alzgraph.common import tqdm, ChatClient, normalize_text, read_json, stable_id, write_json
from alzgraph.metrics import rouge_l, summarize_scores, token_f1
from alzgraph.retrieval import AlzGraphRetriever

SYSTEM = """You are a cognitive neurologist generating a diagnostic impression for a memory-clinic report.
From the patient history, cognitive assessment, and biomarker panel, produce an impression that summarizes:
(1) the cognitive syndrome and severity, (2) the ATN biomarker profile (amyloid, tau, neurodegeneration) and
whether it supports Alzheimer's disease, (3) the most likely diagnosis and stage, and (4) management or
follow-up recommendations. Be concise, evidence-grounded, and clinically safe."""


def build_local_preview(raw_jsonl: str, out_json: str) -> None:
    rows = []
    for line in Path(raw_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        src = json.loads(line)
        rows.append(
            {
                "id": stable_id(line, prefix="t2"),
                "patient_history": normalize_text(src.get("patient_history", "")),
                "cognitive_assessment": src.get("cognitive_assessment", {}),
                "biomarker_panel": src.get("biomarker_panel", {}),
                "gold_impression": normalize_text(src.get("impression", "")),
            }
        )
    write_json(rows, out_json)


def make_prompt(item: dict, retriever: AlzGraphRetriever | None, mode: str) -> list[dict]:
    body = f"""Patient history:
{item.get('patient_history', '')}

Cognitive assessment:
{item.get('cognitive_assessment', {})}

Biomarker panel:
{item.get('biomarker_panel', {})}
"""
    if mode == "graph_rag" and retriever:
        query = " ".join(
            [
                item.get("patient_history", ""),
                " ".join(str(v) for v in item.get("cognitive_assessment", {}).keys()),
                " ".join(str(v) for v in item.get("biomarker_panel", {}).keys()),
            ]
        )
        paths = retriever.retrieve(query)["paths"]
        body = "AlzKG context:\n" + "\n".join(paths) + "\n\n" + body
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def evaluate(args: argparse.Namespace) -> None:
    data = read_json(args.dataset)
    retriever = AlzGraphRetriever(args.triplets) if args.mode == "graph_rag" else None
    client = ChatClient(args.model, temperature=0.3)
    rows = []
    for item in tqdm(data[: args.sample or None]):
        pred = client.complete(make_prompt(item, retriever, args.mode), max_tokens=350)
        gold = item.get("gold_impression", "")
        rows.append(
            {
                "id": item.get("id"),
                "prediction": pred,
                "gold_impression": gold,
                "rouge_l": rouge_l(pred, gold),
                "token_f1": token_f1(pred, gold),
                "mode": args.mode,
            }
        )
    write_json(rows, args.out)
    print(summarize_scores(rows, ["rouge_l", "token_f1"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="AlzBench Task 2: Clinical Report Generation.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--raw_jsonl", required=True)
    build.add_argument("--out", default="data/alzbench/t2/local_preview.json")
    ev = sub.add_parser("eval")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--triplets", default="data/alzkg/triplets.json")
    ev.add_argument("--model", default="medgemma-4b-it")
    ev.add_argument("--mode", choices=["no_rag", "graph_rag"], default="graph_rag")
    ev.add_argument("--sample", type=int, default=0)
    ev.add_argument("--out", default="runs/t2_predictions.json")
    args = parser.parse_args()
    if args.command == "build":
        build_local_preview(args.raw_jsonl, args.out)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
