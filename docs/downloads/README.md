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
  <img alt="Entities: 67" src="https://img.shields.io/badge/seed%20entities-67-0EA5E9?style=flat-square">
  <img alt="Triplets: 107" src="https://img.shields.io/badge/seed%20triplets-107-EC4899?style=flat-square">
  <img alt="Sources: 10" src="https://img.shields.io/badge/ontology%20sources-10-EAB308?style=flat-square">
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
| **AlzKG** | An Alzheimer's knowledge graph over genes, biomarkers, stages, treatments, and outcomes, curated from public ontologies and clinical guidelines, plus a literature-scale PMC builder |
| **Graph-RAG** | Retrieval over graph neighborhoods with personalized-PageRank ranking and serialized, evidence-tiered reasoning paths |
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

The **released seed AlzKG** is curated, guideline-grounded, and fully connected:

| Statistic | Value |
|---|---:|
| Entities | **67** |
| Triplets | **107** |
| Cross-layer triplets | **107 (100%)** |
| Relation types | **10** |
| Source ontologies/guidelines | **10** |

> **Honesty note.** For the curated seed graph, each edge's weight is an **evidence
> tier** (1 = emerging, 2 = established, 3 = guideline/landmark), surfaced in
> serialized paths as `[N tier]`. The literature-scale builder
> (`alzgraph/build_kg.py`) instead populates **true paper counts** from a PMC/PubMed
> corpus and surfaces them as `[N papers]`. The benchmark model-comparison tables
> are produced by running the task runners against an LLM endpoint; this release
> ships the runners and metrics, not third-party model outputs.

Build the seed graph and reproduce its statistics:

```bash
python scripts/build_seed_kg.py
```

Build a literature-scale preview from local PMC XML files:

```bash
python -m alzgraph.build_kg --pmc_dir /path/to/pmc_xml --out_dir data/alzkg
```

Triplets follow the schema:

```json
{
  "head": "APOE", "relation": "pharmacogenomic_consideration", "tail": "Lecanemab",
  "head_layer": "gene", "tail_layer": "treatment",
  "paper_count": 3, "weight_label": "tier", "evidence_tier": 3,
  "evidence": "APOE e4 homozygotes have markedly higher ARIA risk; ...",
  "head_source": "OMIM", "tail_source": "AAN_AA_AUC"
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

# 1) Build the released seed AlzKG (prints real statistics)
python scripts/build_seed_kg.py

# 2) Intrinsic retrieval ablation (no API key needed)
python scripts/retrieval_ablation.py

# 3) Evaluate an LLM with Graph-RAG (needs an API key)
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
    ontology.py      # curated AD 5-layer ontology + relations (evidence-tiered)
    build_kg.py      # literature-scale PMC/PubMed co-occurrence builder
    retrieval.py     # Graph-RAG: PPR neighborhood + path serialization
    metrics.py       # accuracy, ROUGE-L, drug/ARIA safety, KG evidence coverage
    common.py        # IO + OpenRouter-compatible ChatClient
  tasks/             # t1..t5 AlzBench task runners
  scripts/
    build_seed_kg.py       # materializes the released seed AlzKG + demo graph
    retrieval_ablation.py  # intrinsic Graph-RAG ablation (no LLM)
    run_all.sh
  data/
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
