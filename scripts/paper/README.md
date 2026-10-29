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
| `fig03_alternatives.py` | Figure 3 alternatives for the Wasserstein decision (ledger M-42) |
| `participation_ratio.py` | Supplementary figure: effective dimensionality before and after pre-training, by family |
| `umap_protspace_figure.py` | Supplementary UMAP figure (R1.3), from a ProtSpace bundle |
| `prottucker.py` | ProtTucker embeddings. `embedding_generation.py` has no entry for ProtTucker or CLEAN, so without this the two task-specific arms could not be regenerated at all |

## functional/

| script | produces |
|---|---|
| `make_summary.py` | `SUMMARY.md` for the functional axis — the source of the EC/GO-MF numbers in Results |
| `slice_union.py`, `slice_union.sbatch` | slices all 26 embedding arms to the EC+GO union cohort |
| `noise_floor.py` | the empirical noise floor and the hubness behind the `random_1024` validity check |
| `hbi_symmetrisation.py` | the cost of symmetrising the MMseqs2 hit table |
| `duplicate_sequences.py` | the identical-sequence ceiling |

## Top level: the checks behind the October 2026 revision

Each answers one reviewer point or plan task; the outputs are in the private repo's
`manuscript/paper_inputs/<what>_<yyyy-mm-dd>/`, each folder with a README naming the script and commit.

| script | what it measures or makes |
|---|---|
| `_cohort.py` | the shared protein cohort, computed from the native arms (imported by the others) |
| `cut_pca_to_cohort.py`, `rebuild_pca_arm.py` | the PCA arms cut to, or re-projected on, the shared cohort |
| `pair_table_counts.py` | pairs per split that carry each similarity target |
| `pair_redundancy.py` | how unevenly proteins share the test pairs; what a per-protein cap changes |
| `split_homologs.py` | the share of a test set with a close relative among the training proteins |
| `organism_map.py` | the pair set by taxon, length, function and search truncation (R1.3, Table S5) |
| `umap_subsample.py` | UMAP of a superkingdom-stratified subsample, and what a 2-D map keeps (R1.3) |
| `wasserstein_scaling.py` | whether Figure 3's Wasserstein ordering depends on min-max scaling (R1.7, R2.3) |
| `ridge_diff_probe.py` | closed-form ridge on \|a−b\| or (a−b)²: the linear read-out of R1.8 |
| `probe_train_loss.py` | per-epoch train and validation loss of every probe, from the offline wandb runs (R1.10) |
| `rescore_test_predictions.py` | re-scores saved test predictions of a probe grid on subsets of the test pairs |
| `exptm_final_grid.py` | whether Figure 1B's TM-score ranking holds on experimental structures (R2.2) |
| `exptm_and_training_summary.py` | two D14 checks from `probe_metrics.csv` alone |
| `d14_open_collect.py` | the D14 open-decision runs against the main grid (R1.8 read-outs, R2.5 head width) |
| `spearman_facts.py` | re-derives the text's claims in Spearman ρ on the main grid (D14) |
| `new2024_summary.py` | New2024 generalisation per model, with protein-level confidence intervals (C9) |
| `new2024_training_homologs.py` | New2024 proteins with a close homolog among the training proteins |
| `new2024_uniparc_history.py` | New2024 proteins with a public sequence record before UniProt 2024_01 |

## What is deliberately not here

The outputs themselves. These scripts read a results directory and write into it; that
directory is the Zenodo deposit, not the repository.
