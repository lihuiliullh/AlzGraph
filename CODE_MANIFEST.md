# Code Manifest

This repository is the paper-aligned code release for the **AlzGraph / AlzBench**
project: an Alzheimer's disease knowledge graph (AlzKG), a five-task
evidence-intensive reasoning benchmark (AlzBench), and a Graph-RAG retriever.

## Paper-To-Code Mapping

| Paper component | Release code | Notes |
|---|---|---|
| AD corpus collection | `scripts/fetch_pubmed.py`, `scripts/fetch_pmc_fulltext.py` | Find real AD papers via NCBI E-utilities, then download PMC open-access full text and reduce to candidate sentences (≥2 entity mentions); standard library only |
| NER lexicon | `alzgraph/lexicon.py` | AD entity vocabulary + synonyms mapped to canonical entity + layer (case-sensitive gene symbols); merges the ontology-derived `data/lexicon/lexicon_full.json` on top of the hand-curated seed when present. Fixed (paper Sec. 2.5): cross-concept alias merging now requires matching layers and a distinctive (multi-word or ≥5-char) shared surface form, and any `ci` alias ≤3 chars is dropped lexicon-wide |
| Ontology lexicon build | `scripts/build_lexicon_from_ontologies.py` | Compiles the large recognition vocabulary from HGNC, HPO, and ChEBI into `data/lexicon/lexicon_full.json` (raw downloads under `data/lexicon/sources/` are not tracked). Fixed (paper Sec. 2.5): a gene's own primary HGNC symbol is now reserved before alias conflicts are resolved (previously order-dependent — e.g. NANOS1's alias "NOS1" could silently strip the real NOS1 gene's own symbol), and a curated `CLINICAL_ABBREV_BLOCKLIST` drops short gene aliases (and, for a few primary-symbol cases — CBS, NPS, PGC — the primary symbol itself) that collide with a common English word, clinical abbreviation, institutional acronym, or domain-generic term (e.g. bare "tau" as a MAPT alias) |
| AlzKG mining (primary) | `scripts/build_kg_from_fulltext.py` | Sentence-grounded relation extraction over PMC full text: an edge needs a cross-layer entity pair co-occurring in one sentence *and* a relation trigger phrase (curated per-relation templates), with true paper counts; writes `data/alzkg/*` and the demo graph |
| AlzKG mining (abstract alternate) | `scripts/build_kg_from_corpus.py` | Abstract-level cross-layer co-occurrence builder (coarser signal) with true paper counts |
| AlzKG schema + curated seed (optional) | `alzgraph/ontology.py`, `scripts/build_seed_kg.py` | Curated, guideline-tiered five-layer seed graph alternative |
| PMC XML builder (optional) | `alzgraph/build_kg.py` | Co-occurrence builder for local PMC/PubMed XML files |
| Graph-RAG retrieval | `alzgraph/retrieval.py` | Personalized-PageRank neighborhood retrieval and reasoning-path serialization; pure-Python PPR fallback, optional networkx fast path |
| Evaluation metrics | `alzgraph/metrics.py` | Task accuracy, ROUGE-L, Token-F1, BLEU-1, ranking metrics, drug-safety (ARIA/contraindication), guideline concordance, KG evidence coverage |
| T1 Clinical Decision Accuracy | `tasks/t1_clinical_decision_accuracy.py` | AD diagnosis/staging MCQ and open-ended QA |
| T2 Clinical Report Generation | `tasks/t2_clinical_report_generation.py` | Cognitive + fluid/imaging biomarker panel to diagnostic impression; ADNI / memory-clinic data is private, so a local JSONL adapter preserves the evaluation logic. The adapter ships a single synthetic preview case (`data/alzbench/t2/local_preview.json`); run against the three local models as an n=1 illustrative example only, not a benchmark result (paper Sec. 5) |
| T3 Biomarker-Driven Precision Medicine | `tasks/t3_biomarker_precision_medicine.py` | APOE-genotype and biomarker-aware anti-amyloid mAb selection with ARIA safety scoring |
| T4 Treatment Recommendation | `tasks/t4_treatment_recommendation.py` | Dementia-filtered MedQA-USMLE / MMLU builder (needs the `datasets` package) plus treatment safety and KG evidence coverage |
| T4 dataset build (pip-free alternate) | `scripts/build_t4_medqa_subset.py` | Builds the same dementia-filtered MedQA-USMLE subset via the Hugging Face datasets-server HTTP API (`requests` only, no `datasets`/`pip install`); shares the word-boundary term filter in `tasks/t4_treatment_recommendation.py` and applies a small, documented manual-relevance exclusion list on top of it |
| T5 Deep Research Planning | `tasks/t5_deep_research_planning.py` | Builds literature-grounded research-planning instances and evaluates generated study plans |
| KG-only MCQ baseline | `tasks/kg_baseline.py` | Deterministic, no-LLM baseline that answers MCQs from AlzKG evidence alone (lexicon NER + literature-weighted PPR); writes `data/alzkg/kg_baseline_results.json`. Measured on the post-fix graph: T1 = 0.50 (n=20), T3 = 0.30 (n=10), T4 = 0.375 (n=16) accuracy vs. 0.25 random — T4 was exactly at chance (0.25) on the pre-fix graph, the clearest model-facing signal that the Sec. 2.5 lexicon fix helped |
| AlzKG extraction-precision audit | `scripts/audit_kg_sample.py` | Draws a reproducible random sample of triplets (seed=42) and resolves each one's grounding sentence via the recognized-entity index; the correctness judgment itself is manual. Two rounds are recorded in `data/alzkg/extraction_audit.json` (paper Sec. 2.5): pre-fix (4 correct / 8 partial / 18 incorrect) and, after fixing the lexicon bugs found, a fresh post-fix sample (6 correct / 12 partial / 12 incorrect) |
| Flat-retrieval (non-graph) baseline | `scripts/bm25_baseline.py` | Pure-stdlib BM25 over the same corpus sentences as Graph-RAG, evaluated with the identical probe set and gold-coverage criterion as `scripts/retrieval_ablation.py`, for a non-graph comparison point. Result: `data/alzkg/bm25_baseline.json` |
| Key-free model evaluation | `alzgraph/common.py` (`ChatClient`, `manual:` model prefix) | Drop-in replacement for the OpenRouter-backed path: renders the exact prompt, queues it (no gold label) to `runs/manual_llm_queue.json`, and reads an operator-filled `runs/manual_llm_cache.json` so a task runner can be scored without any API key. Used to produce the paper's "Claude (direct)" row; see `data/alzbench/manual_llm_eval_results.json` for the measured summary and caveats |
| Local/open-source model evaluation | `alzgraph/common.py` (`ChatClient`, `--base-url`), all `tasks/t*.py` | Every task's `--base-url` flag points `ChatClient` at any OpenAI-compatible endpoint (no API key needed when the URL isn't OpenRouter's); used to run the full harness against a local Ollama server (`http://localhost:11434/v1/chat/completions`). Also gives local calls a 1536-token floor and a raw-reasoning-trace fallback for thinking-enabled models (e.g. Qwen3), and made `option_letter()` prefer the last explicit "answer is X" phrase over the first bare A–D match (the latter matched the lowercase article "a" in long reasoning traces). Used to produce the paper's Llama-3.2-3B/Gemma3-4B/Qwen3-8B rows; see `data/alzbench/local_llm_eval_results.json` for the measured summary, the models actually available on this server, and the extraction-bug writeup |

## Data Provenance and Honesty Notes

- The released **AlzKG** (`data/alzkg/triplets.json`) is **mined from 7,150 real
  PMC open-access full-text papers** retrieved via NCBI E-utilities (317,606 candidate
  sentences, post-fix lexicon — see below; 361,201 under the earlier, less precise
  lexicon). The post-fix rebuild re-ran entity detection (`alzgraph/lexicon.py`)
  over each previously-fetched candidate sentence's own text rather than
  re-fetching PMC (avoiding a redundant ~7,150-request refetch of identical raw
  text purely to re-apply a lexicon change); `scripts/fetch_pmc_fulltext.py`,
  run fresh, applies the current lexicon identically and is the fully faithful
  reproduction path — its default output path is unaffected by this shortcut.
  Each edge's `paper_count` is the true number of distinct supporting
  papers (surfaced in paths as `[N papers]`); edges with fewer than 3 supporting
  papers are dropped. Relations are **sentence-grounded**: a cross-layer entity pair
  must co-occur in a single sentence that also contains a relation trigger phrase
  (curated per-relation templates) — not whole-document
  co-occurrence. An abstract-level co-occurrence builder
  (`scripts/build_kg_from_corpus.py`) is retained as a coarser alternate.
- An optional curated, guideline-tiered seed graph is available via
  `scripts/build_seed_kg.py` (there `paper_count` carries an evidence tier 1-3).
- Sentence-level co-occurrence + trigger-phrase extraction is not error-free: a
  manual audit of 30 randomly sampled triplets (`scripts/audit_kg_sample.py`,
  `data/alzkg/extraction_audit.json`) found only 4 fully correct, 8 partially
  or loosely supported, and 18 not supported by their own grounding sentence
  (most often a false entity match, e.g. the clinical abbreviation "NOS" for
  the gene NOS1, or amino-acid codes inside a mutation notation for the gene
  MET). We traced these to two lexicon bugs and a set of curatable short-symbol
  collisions, fixed them (`alzgraph/lexicon.py`, `scripts/build_lexicon_from_ontologies.py`),
  rebuilt the graph, and re-audited a fresh sample under the same protocol: 6
  correct / 12 partial / 12 incorrect — a real improvement (60% vs. 40% correct-or-
  partial) that still leaves further failure modes (no relation valence, lab-technique-
  as-biomarker category errors, glossary-list artifacts, unhandled negation) unfixed.
  See paper Sec. 2.5 and the Limitations section.
- The benchmark task builders construct items from curated clinical rules
  (T1 MCQ/QA, T3), a term-filtered external dataset (T4, from MedQA-USMLE), and
  real corpus abstracts (T5). Some Graph-RAG model-evaluation cells (Llama-3.3-70B,
  Qwen-2.5-72B, Mistral) are left unpopulated because they require a
  third-party API key (OpenRouter) this release does not embed; GPT-4o and
  Gemini are omitted from the table entirely for the same reason rather than
  shown as unmeasured placeholders. Four rows *are* measured: "Claude (direct)"
  via the key-free `manual:` `ChatClient` mode (T1/T3 items were authored by
  the same model that answered them; two T4 items were excluded because the
  operator had seen their gold labels during dataset QA), and three local
  open-weight models (Llama-3.2-3B, Gemma3-4B, Qwen3-8B) run against this
  machine's local Ollama server via `--base-url` — not subject to the
  self-authorship caveat, and showing genuinely mixed, sub-ceiling results.

## Differences From Earlier Working Scripts

This release version uses relative paths and command-line arguments, removes any
embedded keys, keeps the ADNI / memory-clinic report task as a private local-data
adapter, aligns the five task names/inputs/metrics with the paper, and keeps each
task runnable independently.
