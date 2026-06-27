#!/usr/bin/env bash
# End-to-end AlzGraph demo: build the seed KG, build benchmark datasets, run the
# retrieval ablation, and (if OPENROUTER_API_KEY is set) evaluate Task 1 + Task 3.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"

MODEL="${MODEL:-openai/gpt-4o}"

echo "==> Building released seed AlzKG"
python3 scripts/build_seed_kg.py

echo "==> Building AlzBench datasets"
python3 tasks/t3_biomarker_precision_medicine.py build --out data/alzbench/t3/bpm_mcq.json
python3 tasks/t5_deep_research_planning.py build \
  --abstracts data/alzbench/t5/abstracts_sample.json \
  --out data/alzbench/t5/research_planning.json

echo "==> Retrieval ablation (no LLM required)"
python3 scripts/retrieval_ablation.py

echo "==> KG-only MCQ baseline (no LLM required)"
python3 tasks/kg_baseline.py --dataset data/alzbench/t1/mcq.json \
  --out runs/t1_kg_baseline.json
python3 tasks/kg_baseline.py --dataset data/alzbench/t3/bpm_mcq.json \
  --question-field clinical_scenario --out runs/t3_kg_baseline.json

if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  echo "==> Task 1 (Graph-RAG) with ${MODEL}"
  python3 tasks/t1_clinical_decision_accuracy.py \
    --dataset data/alzbench/t1/mcq.json --model "${MODEL}" --mode graph_rag \
    --out runs/t1_mcq_graph_rag.json
  echo "==> Task 3 (Graph-RAG) with ${MODEL}"
  python3 tasks/t3_biomarker_precision_medicine.py eval \
    --dataset data/alzbench/t3/bpm_mcq.json --model "${MODEL}" --mode graph_rag \
    --out runs/t3_graph_rag.json
else
  echo "==> OPENROUTER_API_KEY not set; skipping LLM evaluation (framework + KG steps done)."
fi
echo "==> Done."
