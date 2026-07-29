# Mining `feat/ivan-infrastructure` onto main

`feat/ivan-infrastructure` was **never merged** — it sits 5 commits / ~7,800 insertions ahead of a
merge-base (`3214d33`) that main has since moved **85 commits** past. A plain merge would therefore
drag 85 commits of superseded code back in. This branch takes the parts that are still valuable and
leaves the rest.

## Mined (11 modules + 2 shell drivers + 2 test files)

| File | Why it matters |
| --- | --- |
| `src/data_preparation/go_semantic_similarity.py` | **Wang (2007) GO semantic similarity, best-match-average.** The plan calls the GO arm "greenfield `go_report.py`" — it is not greenfield, this exists. Direct input to **C1**. |
| `src/data_preparation/brenda_hfsp_validation.py` | Tests whether HFSP separates curated enzyme classes (beta-lactamase Ambler A/B/C/D). This is **C1's "BRENDA gold-standard" cohort**. |
| `src/data_preparation/pdb_tmscore.py` | SIFTS → RCSB → TMalign pipeline for **experimental** TM-scores. main's `evaluation/pdb_tm_bias.py` is a *library with no CLI and no computation side*, so this fills the gap for **B6 / R2.2**. |
| `src/data_preparation/ecod_homology_pairs.py` | Per-ECOD-group distance densities — structural stratification for **C2**. |
| `src/data_preparation/organism_landscape.py` | Distance distributions by organism group + KS tests. Answers the species-composition half of **R1.3 / C6**. |
| `src/data_preparation/merge_parquet_columns.py` | Plumbing: merges new target columns into the train/val/test splits. Every module above needs it. |
| `src/evaluation/overtraining_analysis.py` | Probe-capacity diagnostics — relevant to **R2.5**. |
| `src/evaluation/classification_eval.py` | AUROC + recall@1FP per CATH level from a precomputed pair table. |
| `src/evaluation/retrieval_metrics.py` | Flat-vector retrieval metrics. **See the reconciliation note below.** |
| `src/visualization/create_retrieval_plots.py` | Retrieval panels. |
| `scripts/download_reference_data.sh`, `scripts/run_ivan_pipeline.sh` | Reference-data fetch + a driver for the above. |
| `tests/test_retrieval_metrics.py` | 8 tests, all pass. |

All are wired into the CLI (`plm data go-similarity`, `plm data pdb-tmscore`, `plm evaluate
classification`, `plm figures retrieval`, …) and every one imports cleanly against today's main.

## Deliberately NOT mined

| File | Reason |
| --- | --- |
| `src/data_preparation/ec_hierarchy_distance.py` | Superseded. main has `evaluation/ec_hierarchy.py` (tested, wired into `ec_report`, BRENDA wildcard convention) and `label_adapters.parse_ec`, which already reads UniProt-style TSVs — the one thing this module offered that looked unique. |
| `tests/test_ec_hierarchy.py` | **Filename collides** with main's, which tests main's implementation. Taking Ivan's would have silently replaced a real test file. |
| `src/evaluation/metrics.py` (modified) | Ivan fixed the `_bootstrap_stat` parallel/sequential control flow. **main already fixed the same bug**, independently and more thoroughly, in `4a0cae5` (+94/−54 since the merge-base). |
| `tests/test_bootstrap_metrics.py` | Tests Ivan's version of that fix and imports `src.evaluation.metrics` — a path that no longer exists after the packaging fix. |
| The other 8 modified files | `distance_computation.py`, `embedding_generation.py`, `random_embeddings.py`, `train.py`, `run_experiments.py`, `create_performance_summary_plots.py`, `test_plot.py`, `docs/todo.md` — main has evolved all of them across 85 commits. Diff them individually if you want a specific change; do not take them wholesale. |
| `docs/superpowers/**` | Ivan's planning docs for this work — internal process notes, not code. Still on his branch if wanted. |

## Reconciliation note — retrieval metrics

`evaluation/recall_fp.py` (main) and `evaluation/retrieval_metrics.py` (mined) **both** compute
recall-at-first-FP, and they do **not** share tie-handling semantics:

- `recall_fp` is canonical for anything that reaches the manuscript — per-query ranking over an
  embedding matrix, the adversarial strict tie-walk of Lin et al. 2023, locked edge cases, a full
  test suite and a barrier spec.
- `retrieval_metrics` takes an already-ranked flat `(distances, labels)` vector, which is the shape
  `classification_eval` needs and which `recall_fp` does not currently expose.

They are not interchangeable without rewriting one of them, which is a statistics decision, not a
mechanical one. **Reconcile before any number from `retrieval_metrics` is reported** — most likely by
adding a flat-vector entry point to `recall_fp` and deleting `retrieval_metrics`.

## Caveats

- This is ~5,000 LOC of **previously unreviewed** code. It imports and the suite is green
  (1104 passed), but only `retrieval_metrics` has tests. Nothing here has been run against real data
  in this session.
- Several modules need reference data that is not in the repo (CAFA annotations, ECOD, SIFTS,
  a TMalign binary). `scripts/download_reference_data.sh` is the intended fetcher — read it before
  running it.
- Ivan's fork still holds the original branch. Nothing was force-pushed and no history was rewritten,
  so his branch is intact and can be re-checked at any time.
