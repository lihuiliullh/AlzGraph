# Code Manifest

This repository is the paper-aligned code release for the **AlzGraph / AlzBench**
project: an Alzheimer's disease knowledge graph (AlzKG), a five-task
evidence-intensive reasoning benchmark (AlzBench), and a Graph-RAG retriever.

## Paper-To-Code Mapping

| Paper component | Release code | Notes |
|---|---|---|
| AD corpus collection | `scripts/fetch_pubmed.py` | Fetches real AD abstracts from PubMed via NCBI E-utilities (standard library only) |
| NER lexicon | `alzgraph/lexicon.py` | AD entity vocabulary + synonyms mapped to canonical entity + layer (case-sensitive gene symbols) |
| AlzKG mining (primary) | `scripts/build_kg_from_corpus.py` | Recognizes entities in abstracts and mines cross-layer co-occurrence relations with true paper counts; writes `data/alzkg/*` and the demo graph |
| AlzKG schema + curated seed (optional) | `alzgraph/ontology.py`, `scripts/build_seed_kg.py` | Curated, guideline-tiered five-layer seed graph alternative |
| PMC XML builder (optional) | `alzgraph/build_kg.py` | Co-occurrence builder for local PMC/PubMed XML files |
| Graph-RAG retrieval | `alzgraph/retrieval.py` | Personalized-PageRank neighborhood retrieval and reasoning-path serialization; pure-Python PPR fallback, optional networkx fast path |
| Evaluation metrics | `alzgraph/metrics.py` | Task accuracy, ROUGE-L, Token-F1, BLEU-1, ranking metrics, drug-safety (ARIA/contraindication), guideline concordance, KG evidence coverage |
| T1 Clinical Decision Accuracy | `tasks/t1_clinical_decision_accuracy.py` | AD diagnosis/staging MCQ and open-ended QA |
| T2 Clinical Report Generation | `tasks/t2_clinical_report_generation.py` | Cognitive + fluid/imaging biomarker panel to diagnostic impression; ADNI / memory-clinic data is private, so a local JSONL adapter preserves the evaluation logic |
| T3 Biomarker-Driven Precision Medicine | `tasks/t3_biomarker_precision_medicine.py` | APOE-genotype and biomarker-aware anti-amyloid mAb selection with ARIA safety scoring |
| T4 Treatment Recommendation | `tasks/t4_treatment_recommendation.py` | Dementia-filtered MedQA-USMLE / MMLU builder plus treatment safety and KG evidence coverage |
| T5 Deep Research Planning | `tasks/t5_deep_research_planning.py` | Builds literature-grounded research-planning instances and evaluates generated study plans |
| KG-only MCQ baseline | `tasks/kg_baseline.py` | Deterministic, no-LLM baseline that answers MCQs from AlzKG evidence alone (lexicon NER + literature-weighted PPR); writes `data/alzkg/kg_baseline_results.json`. Measured: T1 = 0.70, T3 = 0.60 accuracy vs. 0.25 random |

## Data Provenance and Honesty Notes

- The released **AlzKG** (`data/alzkg/triplets.json`) is **mined from 11,654 real
  PubMed abstracts** retrieved via NCBI E-utilities. Each edge's `paper_count` is the
  true number of co-mentioning abstracts (surfaced in paths as `[N papers]`); edges
  with fewer than 5 supporting abstracts are dropped. Relations are induced by
  cross-layer co-occurrence (a coarse but transparent signal); sentence-level
  relation extraction is a documented extension not invoked here.
- An optional curated, guideline-tiered seed graph is available via
  `scripts/build_seed_kg.py` (there `paper_count` carries an evidence tier 1-3).
- The benchmark task builders construct items from curated clinical rules and
  public datasets. The Graph-RAG model-evaluation tables in the paper are
  produced by running `tasks/t*.py` against an LLM endpoint; this release ships
  the runners and metrics, not third-party model outputs.

## Differences From Earlier Working Scripts

This release version uses relative paths and command-line arguments, removes any
embedded keys, keeps the ADNI / memory-clinic report task as a private local-data
adapter, aligns the five task names/inputs/metrics with the paper, and keeps each
task runnable independently.
