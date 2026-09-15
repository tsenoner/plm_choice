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
- Integrity: the patterns reconstruct `counts`, `universe` and `intersection_all` exactly.
  `load_keysets_json` re-derives all three from `patterns` on every read and raises if they
  disagree, so a hand-edit that desynchronises them fails instead of drawing a confident
  wrong figure. Both coverage files were measured on the cluster and committed by hand.

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
# offline, from this freeze (the default source)
plm figures coverage-upset --out out/figures/coverage_upset.png

# after the <=2000 aa cut
plm figures coverage-upset --out out/figures/coverage_upset_cohort2k.png \
    --keysets-json freeze/embedding_key_coverage_cohort2k.json

# rebuild from the HDF5 files (needs cluster access)
plm figures coverage-upset --out <png> --h5-dir <dir of .h5>
```

The `--h5-dir` path caches each key list as a `<stem>.keys.txt` sidecar stamped with
`(size, mtime)`, so a re-scan is instant and a changed `.h5` invalidates its cache.

## `embedding_key_coverage_cohort2k.json` (added 2026-08-06)

Same schema, same cohort, measured **after** the M-12 cut to ≤2000 residues (LRZ job
5734016) — so it is the *post-fix* companion to the file above, not a replacement for it.
Keep both: the pre-fix file is the evidence that the defect existed, this one is the
evidence that it was fixed.

Membership collapses from seven patterns to two: **526,871 of 540,881 (97.41%)** proteins
are in every arm, and the remaining **14,010** are missing only from `clean`/`esm1b` —
ESM-1b's 1022-token positional cap, which CLEAN inherits by construction.

**That shortfall WAS documented-not-fixed; since 2026-09-15 it is fixed** (M-12, Option A):
`embedding_excluded_proteins.json` below removes those 14,010 from every arm, so all fifteen
are scored on the same 526,871 proteins. The earlier policy — keep the 14,010 and footnote the
two short arms — was reversed because the headline claim is a *ranking*, and "two of fifteen
rows were scored on a different test set" is the objection to avoid. The precedent it was
modelled on (`footnote_esm1b_out`, canonical-319) stands: there the loss is 52/319 = 16% of a
small contamination-control set, here it is 2.6% of a 540k comparison cohort. Different cost,
different purpose.

The pre-fix consequence, kept because the pre-fix file above still records it: a pair needs both
proteins, so 97.41% protein coverage was ~94.9% of pairs for those two arms.

`state` records that `esm2_3b` is counted from the completed working copy (542,187 keys
pre-cut), not the stale deposit (435,298).


---

## `embedding_excluded_proteins.json` — the shared-cohort exclusion

The id list that `shared.datasets._load_and_filter_data` subtracts from every arm's key set, so
all 15 arms are scored on **one** cohort. **Committed 2026-09-15** (14,010 ids, 193 KB).

**What the 14,010 are.** Exactly the Swiss-Prot proteins of **1023–2000 residues** — verified two
independent ways that agree with zero symmetric difference: union-minus-intersection over the 15
physical `.h5` key sets, and a pure length cut over `data/raw/sprot_2024/sprot.fasta.fai`. So this
is not a 2.6% trim of a ≤2000 cohort; it **redefines the cohort as ≤1022 residues**, with ESM-1b's
context window setting the limit for all fifteen models. Methods must say so in those words (M-12).
**Consequence: no claim about long or multi-domain proteins is available from this cohort.**

**Why exclusion and not inclusion.** The excluded set is ~38x smaller than the included one
(14,010 vs 526,871 ids), so it is the compact half to commit.

**Why a load-time filter and not deleting datasets.** The `.h5` files are the md5-verified Zenodo
deposit. Deleting from them is irreversible and would make each file stop matching its published
checksum. A filter over a committed id list is reversible, reviewable, and reproducible from the
deposit as published.

**Why it matters.** `shared/datasets.py` drops a pair when *either* protein is missing, so coverage
loss is **quadratic** — `esm2_3b` was scored on 558,947 test pairs where ten other arms got 872,572,
yet is published at rank #10. A cross-pLM ranking whose rows were scored on different data is not a
ranking. **Until this file exists the filter is a no-op**, which is deliberate: adopting the cohort
is an explicit act (commit the freeze), not an accident.

Schema: `schema_version`, `arms`, `counts` (keys per arm), `universe`, `intersection_all`,
`n_excluded`, `content_sha256` (SHA-256 of the sorted id list), `excluded_ids`.

### Regenerate

```bash
plm data cohort-freeze --h5-dir <dir of .h5>            # writes freeze/embedding_excluded_proteins.json
plm data cohort-freeze --h5-dir <dir> --overwrite       # replace an existing freeze
```

Run it against the cohort the analysis actually uses (`embeddings_cohort2k/`, not the raw deposit).
The writer is atomic (`shared.atomic_io.atomic_write`), lands at the canonical path, and **refuses
to clobber an existing freeze unless `--overwrite` is passed** — same contract as the EC freeze
above. `verify_exclusion` re-derives the hash and both counts from `excluded_ids` alone on every
write, so a hand-edited list cannot filter a different cohort than it claims to.

Against `embedding_key_coverage_cohort2k.json` this must produce **14,010** ids
(540,881 − 526,871) — the proteins `clean`/`esm1b` lack to ESM-1b's 1022-token cap. That identity is
pinned by `tests/test_protein_cohort.py::test_the_exclusion_and_coverage_freezes_describe_the_same_cohort`,
the committed artifact by `::test_the_committed_exclusion_freeze_is_the_one_the_paper_describes`, and
the length claim by `::test_the_excluded_proteins_are_exactly_the_ones_longer_than_esm1bs_context`.

On the LRZ login node pass `--core-driver-max-bytes 0`. The default 4.0e9 ceiling loads each arm
into RAM, and fifteen accumulated key sets plus one 3.5 GB arm peak at ~4.1 GB against that node's
4 GiB per-user cgroup. Disabling the core driver costs ~2x on read and drops the peak to 854 MB —
measured 105.7 s end to end for all 15 arms, versus a 4-day queue wait for the `lrz-cpu` batch
equivalent in `scripts/lrz/cohort_freeze.sbatch` (kept for the reproduction record).
