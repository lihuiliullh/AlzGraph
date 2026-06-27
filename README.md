<div align="center">

# AlzGraph

### Building Generalists for Evidence-Intensive Alzheimer's Disease Reasoning in the Wild

**A knowledge-graph-powered benchmark and code release for evaluating whether AI systems can reason across Alzheimer's literature, ATN biomarkers, risk genes, anti-amyloid therapies, and clinical outcomes.**

<p>
  <a href="#how-to-cite"><img alt="Project" src="https://img.shields.io/badge/AlzGraph-AlzKG%20%2B%20AlzBench-4F46E5?style=flat-square"></a>
  <a href="./LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Graph-RAG" src="https://img.shields.io/badge/Graph--RAG-PPR%20%2B%20Paths-7C3AED?style=flat-square">
  <img alt="AlzBench: 5 tasks" src="https://img.shields.io/badge/AlzBench-5%20tasks-14B8A6?style=flat-square">
  <img alt="Abstracts: 11,654" src="https://img.shields.io/badge/PubMed%20abstracts-11%2C654-EAB308?style=flat-square">
  <img alt="Entities: 77" src="https://img.shields.io/badge/entities-77-0EA5E9?style=flat-square">
  <img alt="Triplets: 702" src="https://img.shields.io/badge/triplets-702-EC4899?style=flat-square">
</p>

<h3>5-Layer Alzheimer's Knowledge Graph · 5 Evidence-Intensive Reasoning Tasks · Graph-RAG out of the box</h3>

<p>
  <a href="#why-alzgraph">Why AlzGraph</a> ·
  <a href="#alzkg-the-knowledge-graph">AlzKG</a> ·
  <a href="#alzbench-tasks">Tasks</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#how-to-cite">Cite</a>
</p>

</div>

---

## Why AlzGraph

Modern medical AI is moving from short-form question answering toward
**evidence-intensive clinical reasoning**: connecting literature, mechanisms,
biomarkers, genetics, treatment choices, safety constraints, and patient
outcomes.

Alzheimer's disease (AD) is a demanding testbed for this shift. Correct answers
often depend on multi-hop evidence: a risk gene (e.g. *APOE* ε4) links to an
amyloid biomarker profile, the profile defines a disease stage (MCI due to AD),
the stage gates an anti-amyloid therapy (lecanemab, donanemab), and the therapy
decision is constrained by genotype-dependent **ARIA** risk and contraindications
such as anticoagulation. **AlzGraph** makes these links explicit through an
Alzheimer's knowledge graph and evaluates whether generalist models can use that
evidence in realistic reasoning tasks.

This repository provides a paper-aligned code release for:

| Component | What it gives you |
|---|---|
| **AlzKG** | An Alzheimer's knowledge graph over genes, biomarkers, stages, treatments, and outcomes, **mined from 11,654 PubMed abstracts** via NCBI E-utilities (with a curated-seed option) |
| **Graph-RAG** | Retrieval over graph neighborhoods with personalized-PageRank ranking and serialized, literature-weighted reasoning paths |
| **AlzBench** | Five benchmark tasks spanning diagnosis QA, report generation, biomarker precision medicine, treatment recommendation, and research planning |
| **Metrics** | Task-specific evaluation utilities (accuracy, ROUGE-L, drug/ARIA safety, KG evidence coverage, ...) |
| **Project page** | A GitHub Pages-ready site with an interactive KG explorer |

---

## AlzKG: The Knowledge Graph

AlzKG organizes Alzheimer's evidence into five connected clinical layers and
links them with evidence-grounded, typed relations to enable multi-hop reasoning.

| Layer | Examples | Source ontologies |
|---|---|---|
| **gene** | APP, PSEN1/2, APOE, TREM2, SORL1, ABCA7, BIN1, MAPT | OMIM, ADSP, GWAS Catalog |
| **biomarker** | CSF/plasma Aβ42, p-tau181/217, amyloid PET, tau PET, hippocampal atrophy, NfL, GFAP, MMSE/MoCA/CDR | NIA-AA (ATN), MeSH, HPO |
| **stage** | preclinical AD, MCI due to AD, mild/moderate/severe AD dementia, PCA, lvPPA | NIA-AA, OMIM, UMLS |
| **treatment** | donepezil, rivastigmine, galantamine, memantine, lecanemab, donanemab, aducanumab | ChEBI, AAN/AA AUC |
| **outcome** | cognitive/functional decline, ARIA-E, ARIA-H, amyloid clearance, mortality | HPO, MeSH |

The released **AlzKG** is **mined from 11,654 real PubMed abstracts** (retrieved via NCBI E-utilities):

| Statistic | Value |
|---|---:|
| PubMed abstracts mined | **11,654** |
| Entities | **77** |
| Cross-layer triplets | **702** |
| Relation types | **10** |
| Edge paper count (median / max) | **14 / 1,729** |

> **Honesty note.** Relations are mined by cross-layer **co-occurrence** over real
> abstracts; each edge's `paper_count` is the **true number of supporting abstracts**
> (surfaced in reasoning paths as `[N papers]`), keeping edges with ≥5 supporting
> papers. Entity recognition uses an ontology-derived dictionary lexicon
> (`alzgraph/lexicon.py`). The benchmark model-comparison tables are produced by
> running the task runners against an LLM endpoint; this release ships the runners
> and metrics, not third-party model outputs. A smaller curated, guideline-tiered
> seed graph is also available via `scripts/build_seed_kg.py`.

Reproduce the mined graph from scratch (fetch real abstracts, then mine):

```bash
python scripts/fetch_pubmed.py --max_papers 12000       # download real AD abstracts
python scripts/build_kg_from_corpus.py --min_papers 5   # mine the KG (prints real stats)
```

Mined triplets follow the schema:

```json
{
  "head": "Lecanemab", "relation": "has_outcome", "tail": "ARIA",
  "head_layer": "treatment", "tail_layer": "outcome",
  "paper_count": 178, "weight_label": "papers",
  "head_source": "ChEBI/AAN", "tail_source": "HPO/MeSH",
  "paper_ids": ["38302750", "..."]
}
```

---

## AlzBench Tasks

| Task | Name | What it measures | Main metrics |
|---|---|---|---|
| **T1** | Clinical Decision Accuracy | AD diagnosis, ATN interpretation, staging, genetics MCQ + open QA | Top-1 accuracy, BLEU-1, ROUGE-L, Token-F1 |
| **T2** | Clinical Report Generation | Cognitive + biomarker panel → neurologist-style diagnostic impression | ROUGE-L, Token-F1, report alignment |
| **T3** | Biomarker Precision Medicine | APOE genotype + biomarker status → anti-amyloid mAb selection with ARIA safety | Top-1 accuracy, drug safety score |
| **T4** | Treatment Recommendation | Guideline-consistent dementia therapy under patient constraints | Top-1 accuracy, drug safety, KG evidence coverage |
| **T5** | Deep Research Planning | Literature-grounded AD research question + feasible study plan | ROUGE-L, Token-F1, LLM-as-judge |

---

## Quick Start

```bash
git clone https://github.com/lihuiliullh/AlzGraph.git
cd AlzGraph
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Build the literature-mined AlzKG from scratch (fetch real abstracts + mine)
python scripts/fetch_pubmed.py --max_papers 12000
python scripts/build_kg_from_corpus.py --min_papers 5

# 2) Intrinsic retrieval ablation (no API key needed)
python scripts/retrieval_ablation.py

# 3) KG-only MCQ baseline -- deterministic, no LLM, no API key
#    Answers MCQs from AlzKG evidence alone (PPR over the mined graph).
#    Measured: T1 = 0.70, T3 = 0.60 accuracy (vs. 0.25 random).
python tasks/kg_baseline.py --dataset data/alzbench/t1/mcq.json \
  --out runs/t1_kg_baseline.json
python tasks/kg_baseline.py --dataset data/alzbench/t3/bpm_mcq.json \
  --question-field clinical_scenario --out runs/t3_kg_baseline.json

# 4) Evaluate an LLM with Graph-RAG (needs an API key)
export OPENROUTER_API_KEY="your_key_here"
python tasks/t1_clinical_decision_accuracy.py \
  --dataset data/alzbench/t1/mcq.json \
  --triplets data/alzkg/triplets.json \
  --model openai/gpt-4o --mode graph_rag \
  --out runs/t1_mcq_graph_rag.json
```

Compare against the no-retrieval baseline by switching `--mode no_rag`. For local
models, point `ChatClient` (in `alzgraph/common.py`) at any OpenAI-compatible
local endpoint. The full demo pipeline is wrapped in `scripts/run_all.sh`.

### Task examples

```bash
# T3: biomarker-driven precision medicine (build, then evaluate)
python tasks/t3_biomarker_precision_medicine.py build --out data/alzbench/t3/bpm_mcq.json
python tasks/t3_biomarker_precision_medicine.py eval \
  --dataset data/alzbench/t3/bpm_mcq.json --model openai/gpt-4o --mode graph_rag

# T4: dementia-filtered MedQA-USMLE subset
python tasks/t4_treatment_recommendation.py build --out data/alzbench/t4/medqa_dementia.json --max_items 200

# T2: ADNI / memory-clinic notes are private — use the local adapter
python tasks/t2_clinical_report_generation.py build \
  --raw_jsonl data/private/your_export.jsonl \
  --out data/alzbench/t2/local_preview.json
```

---

## Repository Layout

```text
AlzGraph/
  alzgraph/
    lexicon.py       # AD NER lexicon: entities + synonyms -> canonical + layer
    ontology.py      # curated AD 5-layer ontology + relations (for the seed graph)
    build_kg.py      # PMC/PubMed XML co-occurrence builder (local files)
    retrieval.py     # Graph-RAG: PPR neighborhood + path serialization
    metrics.py       # accuracy, ROUGE-L, drug/ARIA safety, KG evidence coverage
    common.py        # IO + OpenRouter-compatible ChatClient
  tasks/             # t1..t5 AlzBench task runners
  scripts/
    fetch_pubmed.py         # collect real AD abstracts via NCBI E-utilities
    build_kg_from_corpus.py # mine AlzKG (co-occurrence + real paper counts)
    build_seed_kg.py        # optional curated, guideline-tiered seed graph
    retrieval_ablation.py   # intrinsic Graph-RAG ablation (no LLM)
    run_all.sh
  data/
    corpus/          # fetched abstracts (gitignored; reproducible via fetch_pubmed)
    alzkg/           # triplets.json, kg_stats.json, retrieval_ablation.json
    alzbench/        # t1..t5 task datasets
  docs/              # GitHub Pages site + interactive KG explorer
  paper/             # LaTeX source + compiled PDF
  configs/default.json
```

---

## Project Page

A static GitHub Pages site lives in [`docs/`](./docs/) with an interactive KG
explorer over the released seed graph. To publish:

```text
Settings -> Pages -> Deploy from a branch
Branch: main   Folder: /docs
```

---

## How To Cite

If you use AlzGraph, AlzKG, AlzBench, or this code release, please cite:

```bibtex
@article{alzgraph2026,
  title  = {AlzGraph: Building Generalists for Evidence-Intensive Alzheimer's Disease Reasoning in the Wild},
  author = {Liu, Lihui and contributors},
  year   = {2026},
  note   = {Code: https://github.com/lihuiliullh/AlzGraph}
}
```

> Update the author list, affiliation, and arXiv id in `paper/alzgraph.tex` and
> the BibTeX above before release.

---

## License

Released under the [Apache License 2.0](./LICENSE).

<div align="center">

**AlzGraph turns Alzheimer's evidence into graph structure, then tests whether generalist AI systems can reason with it.**

</div>
