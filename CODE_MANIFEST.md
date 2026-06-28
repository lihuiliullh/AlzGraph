# Code Manifest

This repository is the paper-aligned code release for the **AlzGraph / AlzBench**
project: an Alzheimer's disease knowledge graph (AlzKG), a five-task
evidence-intensive reasoning benchmark (AlzBench), and a Graph-RAG retriever.

## Paper-To-Code Mapping

| Paper component | Release code | Notes |
|---|---|---|
| AD corpus collection | `scripts/fetch_pubmed.py`, `scripts/fetch_pmc_fulltext.py` | Find real AD papers via NCBI E-utilities, then download PMC open-access full text and reduce to candidate sentences (≥2 entity mentions); standard library only |
| NER lexicon | `alzgraph/lexicon.py` | AD entity vocabulary + synonyms mapped to canonical entity + layer (case-sensitive gene symbols); merges the ontology-derived `data/lexicon/lexicon_full.json` on top of the hand-curated seed when present |
| Ontology lexicon build | `scripts/build_lexicon_from_ontologies.py` | Compiles the large recognition vocabulary from HGNC, HPO, and ChEBI into `data/lexicon/lexicon_full.json` (raw downloads under `data/lexicon/sources/` are not tracked) |
| AlzKG mining (primary) | `scripts/build_kg_from_fulltext.py` | Sentence-grounded relation extraction over PMC full text: an edge needs a cross-layer entity pair co-occurring in one sentence *and* a relation trigger phrase (per-relation templates adapted from EpiGraph Table 5), with true paper counts; writes `data/alzkg/*` and the demo graph |
| AlzKG mining (abstract alternate) | `scripts/build_kg_from_corpus.py` | Abstract-level cross-layer co-occurrence builder (coarser signal) with true paper counts |
| AlzKG schema + curated seed (optional) | `alzgraph/ontology.py`, `scripts/build_seed_kg.py` | Curated, guideline-tiered five-layer seed graph alternative |
| PMC XML builder (optional) | `alzgraph/build_kg.py` | Co-occurrence builder for local PMC/PubMed XML files |
| Graph-RAG retrieval | `alzgraph/retrieval.py` | Personalized-PageRank neighborhood retrieval and reasoning-path serialization; pure-Python PPR fallback, optional networkx fast path |
| Evaluation metrics | `alzgraph/metrics.py` | Task accuracy, ROUGE-L, Token-F1, BLEU-1, ranking metrics, drug-safety (ARIA/contraindication), guideline concordance, KG evidence coverage |
| T1 Clinical Decision Accuracy | `tasks/t1_clinical_decision_accuracy.py` | AD diagnosis/staging MCQ and open-ended QA |
| T2 Clinical Report Generation | `tasks/t2_clinical_report_generation.py` | Cognitive + fluid/imaging biomarker panel to diagnostic impression; ADNI / memory-clinic data is private, so a local JSONL adapter preserves the evaluation logic |
| T3 Biomarker-Driven Precision Medicine | `tasks/t3_biomarker_precision_medicine.py` | APOE-genotype and biomarker-aware anti-amyloid mAb selection with ARIA safety scoring |
| T4 Treatment Recommendation | `tasks/t4_treatment_recommendation.py` | Dementia-filtered MedQA-USMLE / MMLU builder plus treatment safety and KG evidence coverage |
| T5 Deep Research Planning | `tasks/t5_deep_research_planning.py` | Builds literature-grounded research-planning instances and evaluates generated study plans |
| KG-only MCQ baseline | `tasks/kg_baseline.py` | Deterministic, no-LLM baseline that answers MCQs from AlzKG evidence alone (lexicon NER + literature-weighted PPR); writes `data/alzkg/kg_baseline_results.json`. Measured: T1 = 0.60, T3 = 0.40 accuracy vs. 0.25 random |

## Data Provenance and Honesty Notes

- The released **AlzKG** (`data/alzkg/triplets.json`) is **mined from 7,150 real
  PMC open-access full-text papers** retrieved via NCBI E-utilities (361,201 candidate
  sentences). Each edge's `paper_count` is the true number of distinct supporting
  papers (surfaced in paths as `[N papers]`); edges with fewer than 3 supporting
  papers are dropped. Relations are **sentence-grounded**: a cross-layer entity pair
  must co-occur in a single sentence that also contains a relation trigger phrase
  (per-relation templates adapted from EpiGraph Table 5) — not whole-document
  co-occurrence. An abstract-level co-occurrence builder
  (`scripts/build_kg_from_corpus.py`) is retained as a coarser alternate.
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
