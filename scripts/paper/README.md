# Paper reproduction scripts

The one-off generators behind the figures, the supplementary table and the side
measurements. They were written next to the runs they served and lived, until now, only in
a local results folder — which is how a result becomes unreproducible without anyone
noticing. They are committed verbatim apart from one change: the absolute paths they were
written against are environment variables, and an unset one fails by name.

    export REPO=$(pwd)                       # a checkout of this repository
    export PAPER_ARTEFACTS=/path/to/results  # the measured outputs (Zenodo deposit)
    export MANUSCRIPT_DIR=/path/to/bib_2026  # only needed by the two scripts that read the table back

## figures/

| script | produces |
|---|---|
| `run_fig01_2026-09-21.sh` | Figure 1, from the probe metrics — the invocation, and where the PNG was installed |
| `run_2026-09-21.sh` | Figures 2 and S5 (ridge), Figure 3 inputs, and the divisor-stability number the axis choice rests on |
| `run_ridge.sh`, `run_fingerprint.sh` | the earlier ridge and fingerprint passes, kept because they name the inputs each figure used |
| `build_tableS2.py` | Table S2. Verified 2026-09-21: regenerates the installed table, 93/93 lines |
| `extra_numbers.py` | the pre-training gains and the other derived numbers quoted in Results |
| `compare_fig02.py`, `compare_fig03.py` | random-pairs vs aligner-found population comparison (the 0.43 against 0.77) |
| `redraw_fig03_diverging.py` | Figure 3's diverging colour scale |
| `prottucker.py` | ProtTucker embeddings. `embedding_generation.py` has no entry for ProtTucker or CLEAN, so without this the two task-specific arms could not be regenerated at all |

## functional/

| script | produces |
|---|---|
| `make_summary.py` | `SUMMARY.md` for the functional axis — the source of the EC/GO-MF numbers in Results |
| `slice_union.py`, `slice_union.sbatch` | slices all 26 embedding arms to the EC+GO union cohort |
| `noise_floor.py` | the empirical noise floor and the hubness behind the `random_1024` validity check |
| `hbi_symmetrisation.py` | the cost of symmetrising the MMseqs2 hit table |
| `duplicate_sequences.py` | the identical-sequence ceiling |

## What is deliberately not here

The outputs themselves. These scripts read a results directory and write into it; that
directory is the Zenodo deposit, not the repository.
