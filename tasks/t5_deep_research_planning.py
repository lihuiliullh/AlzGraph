"""AlzBench Task 5: Deep Research Planning.

Evaluates whether models can turn an Alzheimer's research abstract into a
focused, feasible study plan grounded in known gene-biomarker-stage-treatment-
outcome evidence from AlzKG.
"""

import argparse

from alzgraph.common import tqdm, ChatClient, normalize_text, read_json, stable_id, write_json
from alzgraph.metrics import rouge_l, summarize_scores, token_f1
from alzgraph.retrieval import AlzGraphRetriever

SYSTEM = """You are an Alzheimer's disease clinical researcher.
Given a paper abstract, generate:
1. a focused, novel research question,
2. a study design rationale (cohort, biomarkers, endpoints),
3. the required evidence or data and key feasibility considerations.
The plan must be feasible, clinically meaningful, and grounded in known
gene-biomarker-stage-treatment-outcome evidence (ATN framework)."""


def build_from_abstracts(abstracts: str, out: str, max_items: int = 163) -> None:
    src = read_json(abstracts)
    rows = []
    for item in src[:max_items]:
        abstract = item.get("abstract") or item.get("summary") or item.get("text", "")
        if not abstract:
            continue
        rows.append(
            {
                "id": stable_id(item.get("pmc_id", ""), abstract[:100], prefix="t5"),
                "pmc_id": item.get("pmc_id", item.get("id", "")),
                "title": normalize_text(item.get("title", "")),
                "abstract": normalize_text(abstract),
                "expert_research_question": item.get("expert_research_question", ""),
                "expert_plan": item.get("expert_plan", ""),
            }
        )
    write_json(rows, out)


def evaluate(args: argparse.Namespace) -> None:
    data = read_json(args.dataset)
    retriever = AlzGraphRetriever(args.triplets) if args.mode == "graph_rag" else None
    client = ChatClient(args.model, temperature=0.3)
    rows = []
    for item in tqdm(data[: args.sample or None]):
        body = f"Title: {item.get('title', '')}\n\nAbstract:\n{item['abstract']}"
        if retriever:
            paths = retriever.retrieve(item["abstract"])["paths"]
            body = "Established AlzKG evidence paths:\n" + "\n".join(paths) + "\n\n" + body
        pred = client.complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}], max_tokens=700)
        gold = "\n".join([item.get("expert_research_question", ""), item.get("expert_plan", "")]).strip()
        row = {"id": item["id"], "prediction": pred, "mode": args.mode}
        if gold:
            row.update({"rouge_l": rouge_l(pred, gold), "token_f1": token_f1(pred, gold)})
        rows.append(row)
    write_json(rows, args.out)
    print(summarize_scores(rows, ["rouge_l", "token_f1"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="AlzBench Task 5: Deep Research Planning.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--abstracts", required=True)
    build.add_argument("--out", default="data/alzbench/t5/research_planning.json")
    build.add_argument("--max_items", type=int, default=163)
    ev = sub.add_parser("eval")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--triplets", default="data/alzkg/triplets.json")
    ev.add_argument("--model", default="openai/gpt-4o")
    ev.add_argument("--mode", choices=["no_rag", "graph_rag"], default="graph_rag")
    ev.add_argument("--sample", type=int, default=0)
    ev.add_argument("--out", default="runs/t5_predictions.json")
    args = parser.parse_args()
    build_from_abstracts(args.abstracts, args.out, args.max_items) if args.command == "build" else evaluate(args)


if __name__ == "__main__":
    main()
