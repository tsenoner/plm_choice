> ## ⚠ ARCHIVED NOTE — not a live task list
>
> This file came across with the `feat/ivan-infrastructure` mining and describes that
> branch's intent at the time it was written. Several entries below were already wrong
> when they landed and are corrected inline; do not action anything here without
> checking it against the code first.
>
> **Live status lives elsewhere:**
> - `docs/IVAN_BRANCH_MINING.md` — what was actually taken, what was not, and why.
> - `docs/resubmission_plan.html` — the execution plan and open items (Tracks A–E).

# How to choose your pLM

## Main display items
- [ ] tabel over of pLM -> suplementary
- [ ] pairwise comparison (Fig 4 + 5)
- [ ] function correlation


## ToDo
- [ ] corralation plot: scatterplot with x-axis model size and y-axis spearman + shapes differentiate betweeneuc dist andlinear regression
- [ ] (run largest models (aka. ESM-15B, ProtT5-15B?))?

## Ivan Infrastructure (added 2026-03-19)
New scripts for paper revision. See `docs/superpowers/specs/2026-03-19-ivan-infrastructure-design.md` for full spec.

### Ready to run
- [ ] GO semantic similarity (Wang method): `src/data_preparation/go_semantic_similarity.py`
  - Needs: CAFA annotations file, go-basic.obo (auto-downloads)
  - Output: go_wang_mfo/bpo/cco columns in parquet
- [ ] Randomly initialized pLM baseline: `embedding_generation.py --random_init`
  - Run for each model: `uv run python src/data_preparation/embeddings/embedding_generation.py sequences.fasta esm2_650m --random_init`
  - Output: random_init_<model>.h5
- [ ] PDB experimental TM-scores: `src/data_preparation/pdb_tmscore.py`
  - Needs: TMalign binary on PATH
  - Output: tmscore_exp column in parquet

### Evaluation & validation infrastructure (added 2026-03-19, batch 2)
- [ ] ~~EC-number hierarchy distances: `src/data_preparation/ec_hierarchy_distance.py`~~
  - **⚠ CORRECTION — this file does not exist and was never mined in.** It was superseded
    by `src/evaluation/ec_hierarchy.py` + `label_adapters.parse_ec` (see
    `docs/IVAN_BRANCH_MINING.md`, "Not taken"). `evaluation/ec_hierarchy.py` has no
    pairs→columns CLI, so **there is no producer for `ec_dist_*` on this branch**;
    step 3 of `scripts/run_ivan_pipeline.sh` reports and skips. Writing that producer
    is a prerequisite for the EC half of C1 (plan item E8).
- [ ] BRENDA/HFSP validation: `src/data_preparation/brenda_hfsp_validation.py`
  - Auto-fetches enzyme annotations from UniProt REST API
  - Default: beta-lactamases (EC 3.5.2.6), generalizable to any EC
  - Output: JSON validation report + statistics
- [ ] Recall-at-first-false-positive + AUROC: `src/evaluation/retrieval_metrics.py`
  - Standalone functions, also used by classification_eval.py
  - Tests: `tests/test_retrieval_metrics.py` (8 tests, all passing)
  - **⚠ NOT the canonical recall@1FP.** `evaluation/recall_fp.py` is canonical for
    anything reaching the manuscript; these two do not share tie-handling. Output rows
    from `classification_eval` carry a `recall_metric_source` column recording which
    variant produced them. Reconcile before quoting this path in the paper — see
    `docs/IVAN_BRANCH_MINING.md`, "Reconciliation note".
- [ ] SCOP/ECOD classification evaluation: `src/evaluation/classification_eval.py`
  - Needs: classification parquet (protein_id + fold_id/sf_id/fa_id columns)
  - Output: summary table (parquet + CSV) with AUROC + recall per embedding per level

### New scripts (added 2026-03-20)
- [ ] ECOD homology pair density distributions: `src/data_preparation/ecod_homology_pairs.py`
  - Needs: ECOD domain classification file + pairs parquet with distances
  - Output: per-group KDE density plots
- [ ] Organism landscape analysis: `src/data_preparation/organism_landscape.py`
  - Needs: pairs parquet + organism mapping TSV (protein_id, organism_id)
  - Output: organism-bias distribution plots + KS tests
- [ ] Overtraining analysis: `src/evaluation/overtraining_analysis.py`
  - Needs: models directory with training runs (optional: tbparse for TensorBoard curves)
  - Output: overtraining heatmap + summary parquet/CSV
- [ ] ~~Bootstrap metrics regression tests: `tests/test_bootstrap_metrics.py`~~
  - **⚠ CORRECTION — not taken, file does not exist.** It imported the dead `src.evaluation`
    path; its one unique test was ported into `tests/test_metrics.py` instead.

### Code improvements (already applied)
- [x] Vectorized distance computation in distance_computation.py (~2-3x speedup)
- [x] Fixed drop_nulls().len() -> null_count() in distance_computation.py
- [x] Documented magic numbers in run_experiments.py
- [x] Fixed shebang ordering in ecod_homology_pairs.py (`ec_hierarchy_distance.py` was never taken)
- [x] Removed unused goatools dependency (self-contained OBO parser used instead)
- [x] ~~Fixed classification_eval.py import to relative (works outside pytest)~~
  - **⚠ REVERSED — this was backwards.** A *relative* import breaks the documented
    `python src/evaluation/classification_eval.py` invocation (no parent package →
    `ImportError` before argparse runs), which is exactly how the pipeline drives it.
    It is now absolute, like every sibling module in `src/`.
- [x] Documented mmCIF-to-PDB limitation in pdb_tmscore.py
- [x] Added paginated EC download to download_reference_data.sh
- [x] Documented optional tbparse dependency in overtraining_analysis.py
- [x] Fixed train.py early_stopping_patience help text (said 5, actual default 3)
- [x] Fixed PEP 585 type annotations for codebase consistency
- [x] Fixed test_plot.py to skip gracefully when unknown_unknowns not installed
- [x] Fixed _bootstrap_stat() parallel/sequential control flow in metrics.py

### Future (not for this sprint)
- [ ] One-Embedding as 15th benchmark entry (needs per-residue embeddings, ~100-200 GPU-hrs)
  - See: https://github.com/jcoludar/One-Embedding