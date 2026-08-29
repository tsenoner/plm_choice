# Canonical-set freeze (revision plan v3, Phase 0 item 1 + NEW-3)

The pLM comparison is defined over **one frozen protein set**: the canonical 319
(`2024_novelSeqs2.fasta`). This directory holds the version-controlled freeze — the single
source of truth that `evaluation.population.assert_population` and `verify_analysis` assert
inputs against, and from which the frozen pairwise common index is built.

## Files

- **`canonical_set_319.json`** — the freeze manifest (committed). Schema:
  - `canonical_content_sha256` — normalization-invariant hash of the `(id, sequence)` set
    (sorted by id, upper-cased sequence, one `id\tseq` line each). Changes iff the sequence
    *set* changes — not on line-wrap, header text, or record-order reformatting. **This is the
    hash `verify_analysis` asserts.**
  - `raw_file_sha256` — sha256 of the exact frozen FASTA bytes (informational).
  - `ids` — the 319 sorted canonical ids (`assert_population`'s `expected`).
  - `n_pairs` — `C(319, 2) = 50721`; the size the fan-in barrier expects for any pairwise
    artifact built on the frozen index.
  - `esm1b` (NEW-3) — ESM-1b is architecture-capped at 1022 aa, so it embedded only
    **267/319**; the **52** absent ids are all **> 1022 aa** (`missing_len_min` 1023,
    `missing_len_max` 1927) — i.e. the entire gap is the length cap, nothing else.
    `esm1b_paired_policy` is **locked to `"footnote_esm1b_out"`** (co-PI decision, 2026-06-09;
    see below).

The derived **`pair_index_319.parquet`** (50 721 rows, `id_a < id_b`) is *not* committed — it
is regenerated deterministically from `ids` by `evaluation.canonical_set.build_pair_index`
(see "Regenerate") into the gitignored `data/` tree where the DAG runs. The repo pins the
manifest; the parquet is a reproducible build product.

## Provenance

Frozen from `data/2024_novelSeqs2.fasta` on LRZ
(`…/ge94xik2/plm_choice_lrz/data/2024_novelSeqs2.fasta`, raw sha256 `02fb0f36…`,
transfer-verified against the local copy). esm1b coverage read from that project's
`data/2024_new/embeddings/esm1b.h5` (267 keys, a strict subset of the 319, zero foreign ids).

## Regenerate

```bash
python -m evaluation.canonical_set \
    --fasta  <path>/2024_novelSeqs2.fasta \
    --set-name 319 \
    --out-dir <data-dir> \
    --esm1b-h5 <path>/esm1b.h5 \
    --esm1b-paired-policy footnote_esm1b_out \
    --source-uri "lrz:/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0004/ge94xik2/plm_choice_lrz/data/2024_novelSeqs2.fasta"
```

A correct re-run reproduces `canonical_content_sha256 = e27dbdb4…` exactly. The writer is
atomic (`shared.atomic_io.atomic_write`, B7), always lands at the canonical path (never a
timestamped sibling), and **refuses to clobber an existing freeze unless `--overwrite` is
passed** — so a stale freeze can never be silently left behind a "regenerated" one.

## The NEW-3 decision (`esm1b_paired_policy`) — LOCKED

esm1b covers 267 of the 319; the 52 absent are **entirely** the > 1022 aa architecture cap.
The paired-stats policy is a co-PI decision (it changes the Holm denominator and per-cell N).
**Locked 2026-06-09 to `"footnote_esm1b_out"`:**

- ✅ **`"footnote_esm1b_out"`** (chosen) — keep the full **319** for the other 14 pLMs (do not
  discard 52/319 = 16 % of proteins, the long ones, to suit one model). esm1b carries its own
  `n = 267` and is footnoted out of the global paired grid; any direct esm1b-vs-X comparison
  restricts to the common 267 *for that pair only*.
- `"common_267_for_all"` (not chosen) — every pLM scored on the common 267 in all paired
  comparisons; apples-to-apples but drops the 52 long proteins from every cell.

The policy is set reproducibly via `--esm1b-paired-policy` (validated against
`ESM1B_PAIRED_POLICIES`), not a hand-edit. Regardless of policy, no analysis may silently
`dropna()` esm1b into a mixed-cohort mean — `assert_population(..., allow_capped=True)` for
esm1b and report its `n = 267` separately.

---

## `embedding_key_coverage.json` (added 2026-08-05)

Which proteins each pLM embedding set actually covers, for the **sprot_pre2024** cohort —
the manifest behind the coverage UpSet figure.

**Why it is committed.** The `.h5` files are on LRZ/Zenodo, not local, and enumerating ~542k
HDF5 group keys costs **~3 minutes per file** over GPFS (~45 min for all 15). Without this
cache every restyle of the figure would need cluster access. Same philosophy as the pair index
above: **the repo pins the manifest; the image is a reproducible build product.** 2.2 KB.

- `counts` — keys per model. `patterns` — bitmask over `models` → number of proteins with
  exactly that membership. `intersection_all` — present in every arm.
- Integrity: the patterns reconstruct `counts`, `universe` and `intersection_all` exactly
  (asserted when the file is written).

**What it shows.** Only **422,972 of 542,238 (78.00%)** proteins are in *every* arm, in five
tiers: 542,238 complete · 542,237 (one outlier protein) · 540,881 (default `--max_seq_len
2000`) · 526,871 (**ESM-1b's 1022-token cap; CLEAN inherits it**) · 435,298 (`esm2_3b`, an
interrupted run, being completed). Note this is the *same* ESM-1b ceiling documented for the
319 set above — there 267/319, here 526,871/542,238.

This matters because `src/shared/datasets.py:99-105` drops a pair when *either* protein is
missing, so coverage loss is **quadratic**: `esm2_3b` was scored on 558,947 test pairs where
ten other arms got 872,572, yet is published at rank #10.

## Regenerate

```bash
plm figures coverage-upset --out out/figures/coverage_upset.png   # offline, from this freeze
plm figures coverage-upset --h5-dir <dir of .h5>                  # rebuild from the HDF5 files
```

The `--h5-dir` path caches each key list as a `<stem>.keys.txt` sidecar stamped with
`(size, mtime)`, so a re-scan is instant and a changed `.h5` invalidates its cache.
