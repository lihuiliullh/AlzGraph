"""Build the AlzBench T4 dementia-filtered MedQA-USMLE subset without the
``datasets`` package.

``tasks/t4_treatment_recommendation.py::build_medqa_subset`` uses
``datasets.load_dataset``, which requires ``pip install datasets`` and is
unavailable in some release/CI environments. This script fetches the same
public split (GBaker/MedQA-USMLE-4-options, test) row-by-row through the
Hugging Face datasets-server HTTP API using only ``requests`` (already a
project dependency), applies the identical dementia-term filter, and writes
the same schema ``build_medqa_subset`` would have produced.

Usage:
    python scripts/build_t4_medqa_subset.py --out data/alzbench/t4/medqa_dementia_subset.json
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from alzgraph.common import stable_id, write_json
from tasks.t4_treatment_recommendation import matches_dementia_terms

API = "https://datasets-server.huggingface.co/rows"
DATASET = "GBaker/MedQA-USMLE-4-options"
SPLIT = "test"
PAGE_SIZE = 100

# Term-filtering (even word-boundary matching) cannot fully tell "the item is
# about AD/dementia" from "an AD-related word appears somewhere in it". These
# were kept by matches_dementia_terms() but manually reviewed out because the
# AD/dementia term only appears in a background detail or a wrong-answer
# option, while the question itself tests unrelated content:
#   t4_4c3f4db41c66 - MS bladder incontinence Rx; "Rivastigmine" is only a distractor option
#   t4_7e8da2da1d49 - colesevelam side effects; "cognitive impairment" only in a distractor option
#   t4_1159d8a68277 - necrotizing fasciitis management; AD/galantamine only in patient history
#   t4_cd6611c47ebb - anticoagulant/seizure-drug interaction; AD only in patient history
#   t4_1eaa577433ad - Wernicke-Korsakoff thiamine biochemistry; "memory loss" is presentation only
MANUALLY_EXCLUDED_IDS = {
    "t4_4c3f4db41c66",
    "t4_7e8da2da1d49",
    "t4_1159d8a68277",
    "t4_cd6611c47ebb",
    "t4_1eaa577433ad",
}


def fetch_all_rows() -> list[dict]:
    rows = []
    offset = 0
    while True:
        for attempt in range(1, 5):
            resp = requests.get(
                API,
                params={"dataset": DATASET, "config": "default", "split": SPLIT, "offset": offset, "length": PAGE_SIZE},
                timeout=60,
            )
            if resp.status_code == 200:
                break
            time.sleep(min(20, 2**attempt))
        else:
            raise RuntimeError(f"datasets-server request failed at offset={offset}: {resp.status_code} {resp.text[:200]}")
        payload = resp.json()
        page = payload.get("rows", [])
        if not page:
            break
        rows.extend(r["row"] for r in page)
        offset += len(page)
        if len(page) < PAGE_SIZE:
            break
    return rows


def build(out: str, max_items: int = 200) -> None:
    raw_rows = fetch_all_rows()
    rows = []
    for item in raw_rows:
        options_dict = item.get("options", {})
        options = [options_dict.get(letter, "") for letter in ("A", "B", "C", "D")]
        text = f"{item.get('question', '')} {' '.join(options)}".lower()
        if not matches_dementia_terms(text):
            continue
        rows.append(
            {
                "id": stable_id(item["question"], prefix="t4"),
                "source": "MedQA-USMLE",
                "question": item["question"],
                "options": options,
                "correct_answer": item["answer_idx"],
                "answer": item.get("answer", ""),
                "contraindicated": [],
            }
        )
        if len(rows) >= max_items:
            break
    n_term_filtered = len(rows)
    rows = [r for r in rows if r["id"] not in MANUALLY_EXCLUDED_IDS]
    write_json(rows, out)
    print(
        f"fetched {len(raw_rows)} rows from {DATASET}/{SPLIT}, {n_term_filtered} passed the term filter, "
        f"{len(rows)} kept after excluding {n_term_filtered - len(rows)} off-topic items -> {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AlzBench T4 dementia-filtered MedQA-USMLE subset (no pip datasets needed).")
    parser.add_argument("--out", default="data/alzbench/t4/medqa_dementia_subset.json")
    parser.add_argument("--max_items", type=int, default=200)
    args = parser.parse_args()
    build(args.out, args.max_items)


if __name__ == "__main__":
    main()
