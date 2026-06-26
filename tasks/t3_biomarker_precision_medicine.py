"""AlzBench Task 3: Biomarker-Driven Precision Medicine.

Evaluates whether models select the most appropriate Alzheimer's therapy given
APOE genotype, ATN biomarker status, and ARIA / safety constraints. Built from
AAN / Alzheimer's Association appropriate-use criteria (AUC) for anti-amyloid
monoclonal antibodies and AD pharmacology. Reports Top-1 accuracy and a drug
safety score (1.0 iff the selection avoids every contraindicated agent).
"""

import argparse
import random

from alzgraph.common import tqdm, ChatClient, option_letter, read_json, stable_id, write_json
from alzgraph.metrics import accuracy, drug_safety_score
from alzgraph.retrieval import AlzGraphRetriever

RULES = [
    {
        "gene": "APOE e3/e4",
        "context": "amyloid-positive (PET) mild AD dementia, no anticoagulation, <=4 cerebral microhemorrhages on MRI",
        "recommended": "Lecanemab",
        "avoid": [],
        "rationale": "Confirmed amyloid early symptomatic AD without ARIA contraindications is an appropriate anti-amyloid candidate with scheduled MRI monitoring.",
    },
    {
        "gene": "APOE e4/e4",
        "context": "amyloid-positive mild AD dementia on chronic anticoagulation for atrial fibrillation",
        "recommended": "Donepezil",
        "avoid": ["Lecanemab", "Donanemab", "Aducanumab"],
        "rationale": "Concurrent anticoagulation markedly raises ARIA-related hemorrhage risk; anti-amyloid mAbs are generally avoided and symptomatic therapy is appropriate.",
    },
    {
        "gene": "APOE e3/e3",
        "context": "amyloid-positive MCI due to AD with MRI showing >4 cerebral microhemorrhages",
        "recommended": "Donepezil",
        "avoid": ["Lecanemab", "Donanemab", "Aducanumab"],
        "rationale": "More than four microhemorrhages or prior macrohemorrhage are AUC exclusions for anti-amyloid mAbs due to ARIA-H risk.",
    },
    {
        "gene": "APOE e3/e3",
        "context": "moderate-to-severe AD dementia (MMSE 12)",
        "recommended": "Memantine",
        "avoid": ["Lecanemab", "Donanemab"],
        "rationale": "Anti-amyloid mAbs are indicated only for MCI / mild AD dementia; memantine is appropriate for moderate-to-severe disease.",
    },
    {
        "gene": "APOE e3/e4",
        "context": "mild AD dementia with symptomatic bradycardia and sick sinus syndrome",
        "recommended": "Memantine",
        "avoid": ["Donepezil", "Rivastigmine", "Galantamine"],
        "rationale": "Cholinesterase inhibitors increase vagal tone and can worsen bradycardia or cause syncope; memantine avoids cholinergic cardiac effects.",
    },
]

SYSTEM = """You are a clinical neurologist specializing in Alzheimer's disease pharmacotherapy.
Select the single most appropriate treatment from A-D using APOE genotype, ATN biomarker status, and
AAN / Alzheimer's Association appropriate-use-criteria safety reasoning (including ARIA risk).
Return only the option letter."""


def build_dataset(out: str, seed: int = 13) -> None:
    random.seed(seed)
    drug_pool = sorted(
        {d for r in RULES for d in r["avoid"]}
        | {r["recommended"] for r in RULES}
        | {"Galantamine", "Rivastigmine", "Aducanumab", "Memantine"}
    )
    rows = []
    for rule in RULES:
        distractors = [x for x in drug_pool if x != rule["recommended"]]
        options = [rule["recommended"]] + random.sample(distractors, 3)
        random.shuffle(options)
        labels = ["A", "B", "C", "D"]
        rows.append(
            {
                "id": stable_id(rule["gene"], rule["context"], prefix="t3"),
                "gene": rule["gene"],
                "clinical_scenario": f"A patient with {rule['context']} is APOE {rule['gene']}. Which treatment is most appropriate?",
                "options": [f"{label}) {opt}" for label, opt in zip(labels, options)],
                "correct_answer": labels[options.index(rule["recommended"])],
                "recommended": rule["recommended"],
                "avoid": rule["avoid"],
                "rationale": rule["rationale"],
            }
        )
    write_json(rows, out)


def evaluate(args: argparse.Namespace) -> None:
    data = read_json(args.dataset)
    retriever = AlzGraphRetriever(args.triplets) if args.mode == "graph_rag" else None
    client = ChatClient(args.model, temperature=0.0)
    rows = []
    for item in tqdm(data[: args.sample or None]):
        body = item["clinical_scenario"] + "\n" + "\n".join(item["options"])
        if retriever:
            paths = retriever.retrieve(body)["paths"]
            body = "AlzKG reasoning paths:\n" + "\n".join(paths) + "\n\n" + body
        pred = client.complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}], max_tokens=50)
        letter = option_letter(pred)
        selected = ""
        for option in item["options"]:
            if option.startswith(f"{letter})"):
                selected = option.split(")", 1)[1].strip()
        rows.append(
            {
                "id": item["id"],
                "prediction": pred,
                "pred_option": letter,
                "gold_option": item["correct_answer"],
                "drug_safety": drug_safety_score(selected, item.get("avoid", [])),
            }
        )
    write_json(rows, args.out)
    print(
        {
            "top1_accuracy": accuracy([r["pred_option"] for r in rows], [r["gold_option"] for r in rows]),
            "drug_safety": sum(r["drug_safety"] for r in rows) / max(len(rows), 1),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AlzBench Task 3: Biomarker-Driven Precision Medicine.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--out", default="data/alzbench/t3/bpm_mcq.json")
    ev = sub.add_parser("eval")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--triplets", default="data/alzkg/triplets.json")
    ev.add_argument("--model", default="openai/gpt-4o")
    ev.add_argument("--mode", choices=["no_rag", "graph_rag"], default="graph_rag")
    ev.add_argument("--sample", type=int, default=0)
    ev.add_argument("--out", default="runs/t3_predictions.json")
    args = parser.parse_args()
    build_dataset(args.out) if args.command == "build" else evaluate(args)


if __name__ == "__main__":
    main()
