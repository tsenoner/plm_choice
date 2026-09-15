# Which pLM to choose?

[![CI](https://github.com/tsenoner/plm_choice/actions/workflows/ci.yml/badge.svg)](https://github.com/tsenoner/plm_choice/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/data-10.5281%2Fzenodo.17469267-blue.svg)](https://doi.org/10.5281/zenodo.17469267)
[![Preprint](https://img.shields.io/badge/bioRxiv-2025.10.30.685515-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.10.30.685515v1)

Analysis code for a systematic comparison of protein language model (pLM) embeddings.

The question the code answers: **when you pick a pLM, what actually changes?** Each
embedding space is scored against three similarity targets — sequence identity
(`fident`), structure (`alntmscore`) and function (`hfsp`, plus an EC-based axis) —
through four probes of increasing capacity, so that *inherent* geometry (training-free
distance) can be separated from *extractable* information (a trained probe).

## Install

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tsenoner/plm_choice.git
cd plm_choice
uv sync --locked
```

Check the environment:

```bash
uv run plm doctor
```

## The `plm` command

Nearly every analysis step is reachable through one CLI — 56 commands over 54 of
the 58 Python entry points. `plm stages` prints the pipeline in dependency order:

```
1. Cohort   plm data novel-2024      derive the novel-2024 protein cohort
2. Embed    plm embed generate       run each pLM over the cohort FASTA -> HDF5
            plm embed random         the untrained random-vector floor
3. Pairs    plm data merge           merge MMseqs2 + Foldseek hits into a pair table
            plm data split           split the pair table into train/val/test
4. Probes   plm train sweep          train the model-type x embedding x target grid
5. Metrics  plm evaluate run-many    evaluate every trained run
            plm evaluate recall-fp   retrieval read-out (recall to first false positive)
            plm evaluate ec          functional axis: EC hierarchical distance
            plm evaluate aac-floor   amino-acid-composition floor
6. Gate     plm evaluate spec-merge  combine the per-family barrier specs
            plm evaluate barrier     check the artifacts against the spec
7. Figures  plm figures summary      performance-vs-size panels
            plm figures pairwise     embedding-space comparison panels
```

Command groups:

| Group          | What it does                                                          |
| -------------- | --------------------------------------------------------------------- |
| `plm data`     | Build cohorts, pair tables, splits and the frozen manifests            |
| `plm embed`    | Generate, randomise and PCA-reduce embedding matrices                  |
| `plm train`    | Fit probes (`fnn`, `linear`, `linear_distance`, `euclidean`)           |
| `plm evaluate` | The analysis-DAG steps that produce the reviewer-facing numbers        |
| `plm figures`  | Redraw the manuscript panels from computed metrics                     |

`-h` works at every level (`plm -h`, `plm evaluate -h`, `plm evaluate ec -h`).

Nine commands wrap helpers under `scripts/`, which is deliberately not packaged
into the wheel; those work from a git clone and say so clearly if run from an
installed copy. Two entry points remain outside the CLI on purpose: a
fully-seeded one-shot simulation with no options, and the shell scripts that
drive MMseqs2/Foldseek.

The CLI is a thin front-end: each command forwards its arguments to the underlying
module unchanged, so `plm evaluate ec --help` shows that module's own options and its
exit code is the module's own. Every step is also runnable directly, e.g.
`uv run python -m evaluation.ec_report ...`.

**Exit codes** (shared by the CLI and the modules): `0` success, `1` a data-level
failure, `2` an operator or config fault.

## Probes

| Model type        | What it fits                                       |
| ----------------- | -------------------------------------------------- |
| `fnn`             | Feed-forward network on the embedding pair         |
| `linear`          | Linear regression on concatenated embeddings       |
| `linear_distance` | Linear regression on the embedding difference      |
| `euclidean`       | Distance baseline — training-free                  |

Results are organised as:

```
models/<dataset>/<model_type>/<target>/<embedding>/<timestamp>/
├── checkpoints/
├── tensorboard/
└── evaluation_results/
```

## Tests

```bash
uv run pytest -q -m "not slow"   # ~1100 tests, about 75 s
uv run pytest -q -m slow         # statistical coverage simulations (minutes)
```

Two test directories are integration-only and skip by default: `tests/create_embeddings`
needs a GPU plus gated HuggingFace checkpoints (opt in with `PLM_RUN_INTEGRATION=1`),
and `tests/novel_2024` needs the bulk UniRef50 dump.

## Reproduce

```bash
./scripts/reproduce.sh          # what runs, and what it needs
```

The heavy inputs are not in this repository — embeddings and pair tables are archived
on Zenodo. Cite the **concept DOI** [`10.5281/zenodo.17469267`](https://doi.org/10.5281/zenodo.17469267),
which always resolves to the newest version.

## Repository layout

```
src/
├── data_preparation/   cohort assembly, pair tables, embeddings, distances
├── evaluation/         the analysis DAG: *_report, *_barrier_spec, stats, barrier
├── training/           probe models and the training grid
├── visualization/      figure code + shared pLM constants
├── shared/             datasets, experiment paths, atomic IO
└── plm_choice/         the `plm` CLI (wraps the above; contains no analysis logic)
scripts/                analysis helpers; wrapped by the CLI, but only when
                        running from a clone (they are not shipped in the wheel)
tests/                  ~1100 tests
freeze/                 frozen canonical-set manifests (small, version-controlled)
docs/                   specification and analysis notes
```

`data/`, `models/`, `out/` and `notebooks/` are deliberately untracked — they hold bulk
artifacts that live on Zenodo instead.

## Data availability

Embeddings, pair tables and metrics: Zenodo concept DOI
[`10.5281/zenodo.17469267`](https://doi.org/10.5281/zenodo.17469267) — 15 HDF5 embedding
files over 542,299 Swiss-Prot proteins, plus the similarity scores and supplementary
data.

Two caveats worth stating plainly:

- **ESM3-open is not MIT.** Despite the name, the `esm3-sm-open-v1` weights are under
  EvolutionaryScale's Cambrian **Non-Commercial** License. The deposited `esm3_open.h5`
  holds derived embeddings rather than the checkpoint, but the non-commercial terms are
  EvolutionaryScale's and stand regardless of this repository's MIT licence. Pull the
  checkpoint itself from gated HuggingFace; do not redistribute it.
- **ProstT5 is not in the deposit.** It is excluded from the reported figures (see
  `src/visualization/pairwise_embedding_comparison.py`), so the deposit covers the 14
  pLMs the manuscript reports plus the random baseline.

## Citation

If you use this code or data, cite the paper; `CITATION.cff` in this repository carries
the machine-readable metadata and GitHub renders it under "Cite this repository".

> Senoner, T.; Koludarov, I.; Günther, J.; Shehu, A.; Rost, B.; Bromberg, Y.
> *Which pLM to choose?* bioRxiv (2025). https://www.biorxiv.org/content/10.1101/2025.10.30.685515v1

## License

MIT — see [LICENSE](LICENSE). See **Data availability** above for third-party model
terms, which are not covered by this licence.
