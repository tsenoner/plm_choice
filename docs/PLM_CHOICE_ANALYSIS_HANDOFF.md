# pLM-choice revision — analysis-arm handoff & reasoning (for Tobi)

**Author:** Ivan (with Claude as pair), 2026-06-12
**Repo of record:** this repo (`tsenoner/plm_choice`), worked on directly on `main`.
**Purpose of this doc:** capture *all* the reasoning behind the analysis arms we built for the
revision — especially the statistical decisions that aren't obvious from the code — so you can
review it, sanity-check the stats, and know exactly what real data we still need from you.

> **TL;DR of what we need from you** (details in §5):
> 1. **Swiss-Prot per-residue/per-protein embeddings for all ~15 pLMs** — the real cohort. The
>    `canonical-319` set in the repo is a **smoke-test fixture only**; every arm is built
>    cohort-agnostic and is waiting for your Swiss-Prot embeddings to produce paper numbers.
> 2. **pdb-TM arm data**: experimental-PDB structures + a **TMalign experimental-vs-predicted TM
>    table** staged on LRZ. The arm code is on `main` (parked behind the B4 gate); it's purely
>    data-blocked.
> 3. **Run the analysis DAG on LRZ** with the real embeddings (we've kept everything offline and
>    reproducible; the LRZ deploy is by `git pull`).

---

## 1. What these arms are

The revision answers reviewers by scoring each pLM on **biological ground-truth signals**, each as
a separate "arm" with bootstrap confidence intervals and multiple-comparison control. Every arm
follows the same discipline: a frozen cohort, a per-cell barrier spec (so a partial run fails
loud), BCa confidence intervals, and a permutation/Holm correction where a significance claim is
made. All arms are TDD'd and each was put through a fan of adversarial review agents at its design
and implementation boundaries.

### Arm status

| Arm | Question it answers | Status |
|-----|---------------------|--------|
| **recall-FP** | retrieval: recall at first false positive, per pLM | **shipped to `main`** |
| **SNN** | cross-pLM k-NN Jaccard agreement (per-pLM-pair) | **shipped to `main`** |
| **EC** | per-pLM embedding-distance vs EC-functional-distance correlation (τ-b/ρ, CATH-superfamily homology control) | **shipped to `main`** |
| **AAC floor** | the trivial 20-d amino-acid-composition floor each pLM must beat (one-sided Wilcoxon vs floor) | **shipped to `main`** |
| **orphan** | Bromberg-orphan AUROC (Mann-Whitney U) with vertex-BCa CI | **shipped to `main`** |
| **foundation** | generic `stats.vertex_bca_ci` (pluggable-statistic vertex-bootstrap BCa core) that EC/orphan/cross-pLM all bind to | **shipped to `main`** |
| **cross-pLM agreement matrix** | descriptive: how similarly do two pLMs order/scale/associate the same frozen protein pairs? (ρ, R², W₁) | **shipped to `main`** — full arm (compute core U1–U4 + report/CLI + barrier spec + agreement-matrix assembly/Holm) pushed `76c7dae..7a08427`; arm-level fan-reviewed, runs via `python -m evaluation.cross_plm_report` (this doc's focus) |
| **pdb-TM** | predicted-vs-experimental structure (TM) bias per pLM | **code on `main`, parked behind the B4 gate; data-blocked** (B4 sign-off is recorded out-of-band, not in the repo) |

**Important framing (decided):** the cross-pLM arm is a **descriptive Supplementary agreement
matrix**, *not* a "which pLM is best" ranking. The ranking lives in the ground-truth arms
(recall-FP / SNN / EC / pdb-TM), each of which scores a pLM against an *external* target. Cross-pLM
only asks: when two pLMs are handed identically the same protein pairs, do they agree with each
other? See `src/evaluation/cross_plm.py` and the design spec referenced in §6.

---

## 2. The cross-pLM arm in one paragraph

For each unordered pLM-pair and each distance metric, we compute three **symmetric** agreement
statistics on the two pLMs' embedding-distance vectors over one **common frozen pair index**
(so every pLM is scored on identically the same pairs, same order):

- **ρ (Spearman):** do the two pLMs rank-order the pairs the same way?
- **R²:** squared linear association (via the *signed* Pearson-r vertex bootstrap, then the r→R²
  zero-crossing mapping — no in-resample squaring).
- **W₁ (Wasserstein-1), raw and z-scored:** distance between the two pLMs' *marginal* distance
  distributions — do they put the same overall spread of distances on the cohort?

Each cell carries a point estimate + a **vertex (protein) bootstrap BCa CI**. The vertex bootstrap
is load-bearing: both columns of a cell are pLM-dependent vectors induced from the *same* proteins,
so the resample must draw the same proteins for both — which is exactly what the shared
`stats.vertex_bca_ci` core guarantees (it owns the single per-iteration protein draw).

ρ and R² additionally get a permutation p-value and Holm correction. **W₁ does not — and the
reason is the most important thing in this doc.**

---

## 3. Why W₁ has no p-value and is not in the Holm correction (the load-bearing decision)

This was debated, an alternative was implemented-in-spec, **tested**, and **reverted on evidence**.
We're documenting the whole path because the conclusion is non-obvious and a future reader will
otherwise "fix" it back to the wrong thing.

### 3.1 The vacuous first null

The natural permutation null for a cross-pLM statistic is the **symmetric protein-label
permutation** (relabel pLM-B's proteins, recompute). This is correct for ρ and R². But for W₁ it
is **vacuous**: a symmetric row+column relabel only *reorders* a distance matrix's upper-triangle
multiset; it leaves each matrix's **marginal** distance distribution unchanged. W₁ is a function of
those two marginals only, so every permuted W₁ equals the observed → p ≡ 1.0. (`cross_plm.py`
correctly RAISES for W₁ in `cross_plm_permutation_null`.)

### 3.2 The tempting fix — a pair-level two-sample null — and why it's wrong

To bring W₁ into the Holm family, we considered a **two-sample null** that *is* non-vacuous:

- **sign-flip (swap):** for each pair, swap `(da_ij, db_ij)` with probability ½ → two relabelled
  groups → recompute W₁; or
- **pool-and-resplit:** pool both pLMs' pair-distance values and randomly repartition.

Both test H₀ "the two pLMs induce the same marginal distance distribution," and both are
non-vacuous. We preferred sign-flip (it preserves the per-pair pairing). **Then we tested it.**

### 3.3 The simulation that killed it

A distance matrix's upper-triangle entries are **dyadically dependent**: the `m = n(n−1)/2` pair
values are derived from only `n` underlying proteins, so they are not independent. A pair-level
null permutes in **pair-space** (`m` units) while the data's actual randomness lives in
**protein-space** (`n` units). The null distribution is therefore far too narrow → the observed
statistic looks "extreme" almost always → the test rejects a true null far too often.

We measured the H₀-true type-I error rate (nominal 0.05) across regimes — re-runnable in this
repo via `.venv/bin/python scripts/cross_plm_w1_null_simulation.py` (numpy + scipy only):

| Regime (H₀ true: A and B have equal marginal laws) | type-I @ 0.05 | median observed W₁ |
|---|---|---|
| shared geometry, near-identical pLMs | 0.64 | 0.043 |
| shared geometry, separated | 0.68 | 0.130 |
| **independent same-law, n=40 (realistic "two different pLMs" null)** | **0.72** | 0.149 |
| **independent same-law, n=80** | **0.81** | 0.108 |

A correct test rejects ~5% of the time under H₀. This null rejects **64–81%** of the time, and it
gets **worse as n grows** — the unmistakable signature of the pair-space/protein-space mismatch.
This is *not* confined to degenerate near-identical pairs: the realistic regime (two genuinely
different pLMs with the same marginal law, meaningful W₁ ≈ 0.15) is the **worst**. Feeding such
p-values into Holm gives *uncontrolled* family-wise error — Holm assumes valid/conservative inputs
and cannot rescue a type-I rate inflated 13–16× (0.64/0.05 ≈ 13, 0.81/0.05 ≈ 16).

A **vertex-level** null (which would inherit the dyadic structure) has *no clean construction*
here: both matrices are over the *same* vertices, so there is no vertex label that distinguishes
pLM-A from pLM-B to permute.

### 3.4 The decision

**W₁ stays a descriptive distance: point estimate + vertex-BCa CI, no p-value, not in any Holm
family.** Its dyadically-correct uncertainty is already the vertex-BCa CI (the same machinery the
arm uses everywhere). Significance against a null is the wrong question for a descriptive distance
in a Supplementary agreement matrix. The locally-committed compute core already does exactly this.

**If you (Tobi) want a W₁ significance statement anyway,** the only statistically sound route is a
genuine vertex-level resampling null — we couldn't construct one for shared-vertex matrices, and
flag it as open research, not a blocker. Happy to discuss.

---

## 4. The manhattan decision

We added **manhattan** to the distance axis, so every arm's distance grid is now
`{euclidean, cosine, manhattan}`. At the kernel layer this is config-only (`pairwise_distance_long`
already maps `"manhattan" → "cityblock"`). The cross-pLM Holm structure is therefore **6 families**:
`{ρ, R²} × {euclidean, cosine, manhattan}`, each Holm-corrected over the `C(15,2) = 105` pLM-pairs.
W₁ (raw + z) is assembled as a descriptive 15×15 matrix per distance, outside Holm.

One implementation caveat we've recorded for ourselves (so the manhattan column can't silently
vanish): the cross-pLM barrier spec must clone the **SNN** barrier (3-element `DEFAULT_DISTANCES`),
**not** the EC barrier (2-element), and the assembly's distance loop must iterate all three.

---

## 5. What we need from you (Tobi)

1. **Swiss-Prot embeddings for the ~15 pLMs.** Every arm is cohort-agnostic and tested on fixtures;
   `canonical-319` is a smoke test (and `esm1b` is capped at 267/319 there). Paper numbers need your
   real Swiss-Prot embeddings. The arms read per-pLM `.h5` embeddings and a frozen
   `pair_index_<set>.parquet`.
2. **pdb-TM arm data on LRZ:** the experimental-PDB structures + a **TMalign experimental-vs-
   predicted TM table** (`pdb_tm_bias.py` is dataframe-in). This is the only thing blocking the
   pdb-TM run; the arm code is on `main`, parked behind the B4 gate (B4 sign-off is out-of-band,
   not recorded in the repo).
3. **Review + run on LRZ.** Please sanity-check the statistics (especially §3 — we'd value a second
   opinion that "W₁ = descriptive + CI, no p" is the right call), then run the analysis DAG on the
   real embeddings. The LRZ code deploys by `git pull --ff-only` on the login node; data/artifacts
   live outside the clone.

---

## 6. Where the detail lives

- **Code (all on `main`):** `src/evaluation/cross_plm.py` (compute core),
  `cross_plm_report.py` (bridge + CLI), `cross_plm_barrier_spec.py` (fan-in barrier),
  `cross_plm_matrix.py` (agreement-matrix assembly + Holm over the 6 families).
- **Full design spec** (with the W₁ reasoning and build units): the companion PEE repo,
  `docs/superpowers/specs/2026-06-11-cross-plm-design.md`, §9 = the 2026-06-12 revision.
- **The simulation** behind §3.3: `scripts/cross_plm_w1_null_simulation.py` (this repo).
  Re-runnable with numpy + scipy; it prints the table above.
- **Shared stats core:** `src/evaluation/stats.py` — `vertex_bca_ci` (the generic vertex-bootstrap
  BCa), `wasserstein_w1`, `holm_bonferroni`, `_r2_from_r_ci`.

Questions on any of the statistical calls → ask; the reasoning above is exactly why each was made.
