# Code Manifest

This repository is the paper-aligned code release for the **AlzGraph / AlzBench**
project: an Alzheimer's disease knowledge graph (AlzKG), a five-task
evidence-intensive reasoning benchmark (AlzBench), and a Graph-RAG retriever.

## Paper-To-Code Mapping

| Paper component | Release code | Notes |
|---|---|---|
| AlzKG five-layer schema and curated ontology seed | `alzgraph/ontology.py` | Layers: gene, biomarker, stage, treatment, outcome; curated cross-layer relations with evidence tiers and source provenance |
| AlzKG construction from literature | `alzgraph/build_kg.py` | Lightweight reproducible co-occurrence builder for PMC/PubMed XML; populates true literature paper counts |
| Released seed AlzKG builder | `scripts/build_seed_kg.py` | Materializes `data/alzkg/triplets.json`, `paper_metadata.json`, and the project-page demo graph from the curated ontology; prints real graph statistics |
| Graph-RAG retrieval | `alzgraph/retrieval.py` | Personalized-PageRank neighborhood retrieval and reasoning-path serialization; pure-Python PPR fallback, optional networkx fast path |
| Evaluation metrics | `alzgraph/metrics.py` | Task accuracy, ROUGE-L, Token-F1, BLEU-1, ranking metrics, drug-safety (ARIA/contraindication), guideline concordance, KG evidence coverage |
| T1 Clinical Decision Accuracy | `tasks/t1_clinical_decision_accuracy.py` | AD diagnosis/staging MCQ and open-ended QA |
| T2 Clinical Report Generation | `tasks/t2_clinical_report_generation.py` | Cognitive + fluid/imaging biomarker panel to diagnostic impression; ADNI / memory-clinic data is private, so a local JSONL adapter preserves the evaluation logic |
| T3 Biomarker-Driven Precision Medicine | `tasks/t3_biomarker_precision_medicine.py` | APOE-genotype and biomarker-aware anti-amyloid mAb selection with ARIA safety scoring |
| T4 Treatment Recommendation | `tasks/t4_treatment_recommendation.py` | Dementia-filtered MedQA-USMLE / MMLU builder plus treatment safety and KG evidence coverage |
| T5 Deep Research Planning | `tasks/t5_deep_research_planning.py` | Builds literature-grounded research-planning instances and evaluates generated study plans |

## Data Provenance and Honesty Notes

- The released **seed AlzKG** (`data/alzkg/triplets.json`) is curated from public
  ontologies and clinical guidelines. For seed edges, `paper_count` carries an
  **evidence tier** (1 = emerging, 2 = established, 3 = guideline/landmark),
  surfaced in serialized paths as `[N tier]`. The `build_kg.py` PMC pipeline
  instead populates true literature paper counts, surfaced as `[N papers]`.
- The benchmark task builders construct items from curated clinical rules and
  public datasets. The Graph-RAG model-evaluation tables in the paper are
  produced by running `tasks/t*.py` against an LLM endpoint; this release ships
  the runners and metrics, not third-party model outputs.

## Differences From Earlier Working Scripts

This release version uses relative paths and command-line arguments, removes any
embedded keys, keeps the ADNI / memory-clinic report task as a private local-data
adapter, aligns the five task names/inputs/metrics with the paper, and keeps each
task runnable independently.
