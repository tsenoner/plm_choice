# Mining `feat/ivan-infrastructure` onto main

`feat/ivan-infrastructure` (tip `cbf984a`) was **never merged**. It is 5 commits /
7,801 insertions ahead of merge-base `3214d33`, which main has since moved **85 commits**
past. A plain merge would therefore drag 85 commits of superseded code back in — including
a revert of two deliberate statistics fixes. So it was mined file by file instead.

**Every one of the 28 files is accounted for below.** A first pass made a blanket
"main evolved these, skip" call on the nine modified files; that was wrong, and the
second pass replaced it with per-file three-way diffs (`merge-base→Ivan`,
`merge-base→main`, `main→Ivan`). Four of those nine turned out to be files **main never
touched**, so there was no conflict to avoid — only unreviewed work being left behind.

## Taken

### New modules (11 + 2 shell drivers)

| File | Why |
| --- | --- |
| `data_preparation/go_semantic_similarity.py` | **Wang (2007) GO semantic similarity, best-match-average.** The plan calls the GO arm "greenfield `go_report.py`" — it is not greenfield. Direct input to **C1**. |
| `data_preparation/brenda_hfsp_validation.py` | Tests whether HFSP separates curated enzyme classes (beta-lactamase Ambler A/B/C/D). **C1's BRENDA gold-standard cohort.** |
| `data_preparation/pdb_tmscore.py` | SIFTS → RCSB → TMalign pipeline for **experimental** TM-scores. main's `evaluation/pdb_tm_bias.py` is a library with no CLI and no computation side, so **B6 / R2.2** had no way to produce its input. |
| `data_preparation/ecod_homology_pairs.py` | Per-ECOD-group distance densities — structural stratification for **C2**. |
| `data_preparation/organism_landscape.py` | Distance distributions by organism group + KS tests — the species-composition half of **R1.3 / C6**. |
| `data_preparation/merge_parquet_columns.py` | Plumbing: merges new target columns into the splits. Everything above needs it. |
| `evaluation/overtraining_analysis.py` | Probe-capacity diagnostics — **R2.5**. |
| `evaluation/classification_eval.py` | AUROC + recall@1FP per CATH level from a precomputed pair table. |
| `evaluation/retrieval_metrics.py` | Flat-vector retrieval metrics. **See the reconciliation note.** |
| `visualization/create_retrieval_plots.py` | Retrieval panels. |
| `scripts/download_reference_data.sh`, `scripts/run_ivan_pipeline.sh` | Reference-data fetch + a driver. |
| `tests/test_retrieval_metrics.py` | 8 tests, all pass. |

### Changes to existing files (4 files main never touched, + 2 hunk-level takes)

| File | Taken | Not taken |
| --- | --- | --- |
| `embeddings/embedding_generation.py` | **`--random_init` / `--random_seed`** — the frozen randomly-initialised transformer of **B4 / C3 / D-6**. main had no equivalent (`grep random_init` → nothing). | Its `_reinit_weights` **was scientifically invalid** and had to be rewritten — see below. |
| `embeddings/random_embeddings.py` | The docstring distinguishing i.i.d. noise from an untrained *architecture*. | — |
| `training/train.py` | Removal of the `choices=["fident","alntmscore","hfsp"]` whitelist on `--param_name`; `--early_stopping_patience` help said 5 against a default of 3. | The `from src.` imports — stale against the packaging fix. |
| `training/run_experiments.py` | Same whitelist removal on `--target_params`, plus the hyperparameter documentation block. | Same stale import. |
| `data_preparation/distance_computation.py` | Two `drop_nulls().len()` → `len(...) - null_count()` (avoids materialising a filtered column). | The vectorised batch distance: a real ~2–3× win, but Euclidean-only, and taking the file would delete main's tested cosine/manhattan/NaN kernel from `92e5c48`. Port separately if runtime matters. |
| `visualization/create_performance_summary_plots.py` | **`--delta_baseline`** — metrics as `(current − baseline)` on a zero-centred axis. This is exactly **D1**'s "redo Sup-Fig-4 as a diff-to-Fig-1, not a duplicate". Ported by hand, plus a warning when the join drops rows. | Ivan's version also re-inlines the pLM constants that were just extracted to `plm_constants.py`. |
| `tests/test_metrics.py` | Ivan's parallel-failure fallback test, adapted to main's signature — main fixed the two paths' reproducibility but never tested that the `except` branch engages. | The rest of `test_bootstrap_metrics.py` (tests Ivan's version of a fix main made differently, and imports the dead `src.evaluation` path). |

### ⚠ The `--random_init` weight scheme had to be replaced

Ivan's `_reinit_weights` applied `N(0, 0.02)` to **every** parameter. That is not ESM's
scheme despite the docstring: `EsmPreTrainedModel._init_weights` gives Linear/Embedding
weights `N(0, 0.02)` but **biases 0 and LayerNorm gains 1.0**. Measured on a 4-layer ESM
config, same seed:

| | Ivan | correct |
| --- | --- | --- |
| LayerNorm.weight mean | 0.0004 | 1.0000 |
| bias abs-mean | 0.0157 | 0.0000 |
| hidden abs-mean | 2.4e-02 | 8.0e-01 |
| residue-residue corr | **+0.960** | +0.686 |

At r = 0.96 every residue vector is nearly identical, so the mean-pooled protein
embedding degenerates to a constant plus noise — the opposite of the "contextual
architecture-only prior" the baseline exists to provide, in the figure R1.9 specifically
asked for.

For HuggingFace models `model_class(model_config)` **already** performs the correct init
via `post_init()`, so the re-init was destroying a correct initialisation; it is now just
`torch.manual_seed(random_seed)` before construction. The native-ESM loader has no
`from_config`, so it keeps a **type-aware** re-init. `tests/test_random_init_baseline.py`
pins the statistics.

## Not taken

| File | Reason |
| --- | --- |
| `data_preparation/ec_hierarchy_distance.py` | Superseded — main has `evaluation/ec_hierarchy.py` (tested, wired into `ec_report`) and `label_adapters.parse_ec`, which already reads UniProt-style TSVs. |
| `tests/test_ec_hierarchy.py` | **Filename collides** with main's, which tests main's implementation. |
| `evaluation/metrics.py` | Ivan's sentinel refactor is **already on main verbatim** (`4a0cae5`, arrived independently), and his header claims a bug that does not exist at the merge-base. Taking the file would revert `_r2_ci_from_r_bounds`, drop the `seed` parameter, and reintroduce the global-RNG mutation main fixed — and break 4 tests on signatures that no longer match. |
| `tests/test_bootstrap_metrics.py` | Imports the dead `src.evaluation` path; its one unique test was ported into `tests/test_metrics.py` instead. |
| `tests/test_plot.py` | Ivan's version merely `skipif`s the broken import; main's F2 fix points it at the real target and adds a second case. |
| `docs/todo.md` | Taken (main never touched it) — but treat it as an archived note, not a live list. |
| `docs/superpowers/**` | Ivan's own planning docs for this work. Process notes, not code; still on his branch. |

## Reconciliation note — retrieval metrics

`evaluation/recall_fp.py` (main) and `evaluation/retrieval_metrics.py` (mined) **both**
compute recall-at-first-FP and do **not** share tie-handling:

- `recall_fp` is canonical for anything reaching the manuscript — per-query ranking over
  an embedding matrix, the adversarial strict tie-walk of Lin et al. 2023, locked edge
  cases, a test suite and a barrier spec.
- `retrieval_metrics` takes an already-ranked flat `(distances, labels)` vector, the shape
  `classification_eval` needs and which `recall_fp` does not expose.

Not interchangeable without rewriting one of them, which is a statistics decision.
**Reconcile before any number from `retrieval_metrics` is reported** — most likely by
adding a flat-vector entry point to `recall_fp` and deleting this module.

## Caveats

- ~5,400 LOC of **previously unreviewed** code. It imports, the suite is green
  (1,141 passed), and everything is reachable from the CLI — but only `retrieval_metrics`
  and the random-init scheme have tests, and none of it has been run against real data.
- Several modules need reference data not in the repo (CAFA annotations, ECOD, SIFTS, a
  TMalign binary). `scripts/download_reference_data.sh` is the intended fetcher — read it
  before running it.
- `pdb_tmscore`'s `--resolution_cutoff` is accepted but **not applied**, and its
  `EXPERIMENTAL_METHODS` allowlist is unused; both now say so. Do not describe that
  pipeline as resolution- or method-filtered until they are wired up.
- Ivan's fork still holds the original branch. Nothing was force-pushed and no history was
  rewritten, so it is intact and can be re-checked at any time.
