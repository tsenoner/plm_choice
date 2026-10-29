# Data dictionary

Formats of every file in this record (version 2). Protein identifiers are UniProt accessions throughout.
Parquet files read with pandas or pyarrow, `.h5` with h5py, `.npz` with `numpy.load`. Paths inside the
`.tar` archives are given as they unpack.

Target names used everywhere: `fident` = PIDE, the fractional sequence identity of the MMseqs2 alignment
(0–1); `hfsp` = HFSP score from that alignment; `alntmscore` = TM-score of the Foldseek alignment of the
two AlphaFold DB v4 models (0–1). A missing value means the search reported no reliable alignment for that
measure.

## Embeddings: `<model>.h5` (15 files)

One HDF5 dataset per protein, named by accession, holding the mean over per-residue final-layer
representations.

| file | model | dim | dtype | stored shape | proteins |
|---|---|---:|---|---|---:|
| `ankh_base.h5` | Ankh base | 768 | float32 | (768,) | 540,881 |
| `ankh_large.h5` | Ankh large | 1536 | float32 | (1536,) | 540,881 |
| `clean.h5` | CLEAN | 128 | float32 | (128,) | 526,871 |
| `esm1b.h5` | ESM1b | 1280 | float32 | (1280,) | 526,871 |
| `esm2_8m.h5` | ESM2 8M | 320 | float32 | (320,) | 540,881 |
| `esm2_35m.h5` | ESM2 35M | 480 | float32 | (480,) | 540,881 |
| `esm2_150m.h5` | ESM2 150M | 640 | float32 | (640,) | 540,881 |
| `esm2_650m.h5` | ESM2 650M | 1280 | float32 | (1280,) | 540,881 |
| `esm2_3b.h5` | ESM2 3B | 2560 | float32 | (2560,) | 540,881 |
| `esm3_open.h5` | ESM3 open | 1536 | float32 | (1536,) | 540,881 |
| `esmc_300m.h5` | ESM C 300M | 960 | float32 | (960,) | 540,881 |
| `esmc_600m.h5` | ESM C 600M | 1152 | float32 | (1152,) | 540,881 |
| `prott5.h5` | ProtT5 | 1024 | float16 | (1, 1024) | 540,881 |
| `prottucker.h5` | ProtTucker | 128 | float16 | (1, 128) | 540,881 |
| `random_1024.h5` | random Gaussian control | 1024 | float16 | (1024,) | 540,881 |

Flatten with `.ravel()` and cast to float32 before computing distances. The evaluation cohort is the
526,871 accessions present in all 15 files (the proteins of at most 1022 residues).

## `pair_tables_splits.tar`

`sprot_pre2024_e1/sets/{train,val,test}.parquet`, one row per protein pair, each unordered pair once.

| column | type | meaning |
|---|---|---|
| `query`, `target` | string | the two accessions |
| `fident` | float64 | PIDE (missing where no reliable sequence alignment) |
| `hfsp` | float64 | HFSP (missing where no reliable sequence alignment) |
| `alntmscore` | float64 | TM-score (missing where no reliable structural alignment) |

Rows: train 59,587,201; validation 8,196,507; test 8,066,264 (75,849,972 in all). The files hold every
pair; the paper's cohort filter (both proteins in all 15 embedding files) removes 940,833 when they are
loaded.

## `pair_table_counts.tar`

`pair_table_counts/sprot_pre2024_e1.json`: pairs per split and how many carry each target (`per_split`,
`total`).

## `identical_sequence_mask.tar`

`ridge_identical/`: pairs whose two proteins have identical sequences (325,019 of 75,849,972). Identical
inputs give identical embeddings, so these pairs are excluded from the distance distributions.

| file | contents |
|---|---|
| `{train,val,test}_identical.parquet` | one boolean column `identical`, row-aligned with the split's pair table |
| `{train,val,test}_identical_keyed.parquet` | `query`, `target` of the identical pairs only |
| `identical_pairs.json` | counts per split and in total |

## Probe grids: `probes_full100_p10.tar`, `probes_pca_cohort.tar`, `probes_randinit_full100.tar`

Layout: `<grid>/<read-out>/<target>/<arm>/`, with `<grid>` = `full100_p10` (native embeddings),
`pca_cohort` (128-dimensional PCA projections) or `randinit_full100` (untrained networks, arms named
`random_init_<model>_seed0`); `<read-out>` = `fnn` (trained probe) or `euclidean` (raw distance, nothing
fitted); `<target>` = `fident`, `hfsp` or `alntmscore`. 15 arms per grid (11 for `randinit_full100`), so
90 cells per grid (66).

| file in a cell | contents |
|---|---|
| `hparams.yaml` | run settings (paths in it point to the cluster's scratch space) |
| `checkpoints/best-epoch=…-val_loss=….ckpt` | PyTorch Lightning checkpoint of the best epoch (`fnn` only) |
| `evaluation_results/*_metrics.txt` | test-set MSE, RMSE, MAE, R², Pearson r, Spearman ρ, standard errors and 95% CIs |
| `evaluation_results/*_predictions_targets.npz` | `predictions`, `targets` (float32), one value per scored test pair |
| `evaluation_results/*_results.png` | predicted against true values |
| `training_complete.txt`, `wandb_run_id.txt` | completion marker; id of the (not included) training log |

The paper's score is the Spearman ρ in `*_metrics.txt`; for the `euclidean` read-out, its magnitude
(distance falls as similarity rises).

## `functional_annotation.tar`

`functional_annotation/`: leave-one-out 1-nearest-neighbour annotation transfer (goPredSim protocol) for
15 pretrained embedding sets, 11 untrained controls and two MMseqs2 homology baselines.

| file | contents |
|---|---|
| `ec_labels_v2.tsv` | `Entry`, `EC number`: experimental (ECO:0000269) EC numbers of Swiss-Prot proteins |
| `ec_strat_v2_freeze.json` | the EC cohort: `ids` (6,902 accessions), `seed`, the stratification (CATH superfamily × EC class × length quartile, at most 25 per cell) |
| `go_mf_labels.tsv` | `protein_id`, `GO_term`: experimental GO molecular-function terms |
| `go_mf_cohort_freeze.json` | the GO-MF cohort: `ids` (20,073 accessions), `seed`, stratification, cap |
| `go-basic.obo` | the Gene Ontology release used to propagate terms |
| `union.fasta`, `union_ids.txt` | sequences and accessions of the 23,336 proteins in either cohort |
| `union_allvsall.m8` | MMseqs2 all-against-all search of the union, tab-separated, no header: `query`, `target`, `fident`, `evalue`, `alnlen`, `qcov`, `tcov` (self hits removed) |
| `union_allvsall.summary.json` | command, MMseqs2 version, hit counts |
| `ec_v2_transfer/`, `go_mf_transfer/` | the transfer runs (below) |
| `go_mf_transfer_noPB/` | GO-MF with "protein binding" dropped, a sensitivity run |
| `ec_tau_b_v2_euclidean.csv`, `go_mf_transfer/tau_b.csv` | Kendall τ-b over all cohort pairs, per arm: embedding distance against EC hierarchical distance (EC) or against 1 − Wang best-match average (GO-MF) |

Each transfer run directory holds:

| file | contents |
|---|---|
| `summary.csv` | one row per `arm` × `distance` (euclidean, cosine) × `variant` × `subset` × `score`: `mean` with bootstrap `ci_lo`, `ci_hi` (2,000 resamples), `chance`, `oracle`, query and tie counts |
| `paired_differences.csv` | per-query paired differences between two arms (`arm_a`, `arm_b`, `mean_diff`, `ci_lo`, `ci_hi`, `excludes_zero`) |
| `per_query.parquet` | one row per arm × distance × variant × query: `neighbour_id`, `neighbour_distance`, tie flags, and the scores (EC: `score_exact`, `score_share3`, `score_share2`, `score_class`; GO-MF: `score_f1`, `score_wang_bma`) |
| `per_query_baseline.parquet` | per query: eligible neighbours, the chance and oracle score, subset membership |
| `manifest.json` | inputs with sha256, parameters, code commit and library versions |

Variants restrict the eligible neighbours: `all`; `fident_lt_0.3` (only neighbours below 30% identity to the
query); `no_hit_evalue_0.001` (only neighbours MMseqs2 does not find at E ≤ 10⁻³). Subsets: `all_queries`,
`hbi_answerable` (queries with a homology hit), `no_hit` (queries without one). Arms `hbi_evalue` and
`hbi_fident` are the homology baselines, which transfer from the query's top eligible MMseqs2 hit ranked by
E-value or by identity. The EC score in the
paper is `exact` (all four EC fields agree), the GO-MF score `f1` (protein-centric F1 of the ancestor-closed
term sets, root excluded).

## `experimental_tm.tar`

`experimental_tm/`: test pairs where both proteins have an experimental PDB chain (one representative chain
per protein from residue-level SIFTS, resolution ≤ 3.0 Å), aligned Cα-only with US-align.

| file | contents |
|---|---|
| `tmscore_exp.parquet` | 10,740 pairs: chains used (`q_pdb`, `q_chain`, method, resolution, mapped length, Cα count, the same with `t_`), `tm1`, `tm2` (normalised by each chain), `tmscore_exp` (by the longer chain), `tmscore_exp_short` (by the shorter), `tmscore_exp_avg`, `rmsd`, `seqid_ali`, `len1`, `len2`, `len_ali`, and the pair's `fident`, `hfsp`, `alntmscore` |
| `exptm_matched.parquet` | `query`, `target`, `alntmscore` = experimental TM-score normalised by the shorter chain (10,740 pairs) |
| `exptm_wholechain.parquet` | the same, normalised by the longer chain (10,740) |
| `exptm_afdbtarget.parquet` | the same pairs with the AlphaFold-based training target (10,400 that have one) |

The three `exptm_*` files share the pair-table format, so the probes score on them unchanged.

## `distance_histograms.tar`

`distance_histograms/random_pairs/` (Figure 2: the 5,000,000 random pairs) and
`distance_histograms/aligner_pairs/` (Table S4 and the Supplement's aligner-found distributions: the
75,849,972 aligner-found pairs, identical sequences excluded), one pair of files per model:

| file | contents |
|---|---|
| `<model>.json` | `n_used`, min, max, mean, sd, quartiles, IQR, quartile coefficient of dispersion (`qcd`), 27 exact percentiles (`quantiles`), histogram ranges, Hartigan's dip test (`dip`) |
| `<model>.npz` | `hist`, `edges`: 50,000-bin histogram of the Euclidean distance; `hist_log10`, `edges_log10`: 5,000 bins of log10 distance; `subsample`: 500,000 distances drawn at random (float32); `quantile_levels`, `quantile_values` |

Every normalization in the paper (min–max, division by a percentile or the median) is an affine map of
these raw-scale histograms. The JSON field `source` records where the distances were read on the
authors' machines.

## `random_pairs.tar`

`random_pairs/`:

| file | contents |
|---|---|
| `random_pairs_distances.parquet` | 5,000,000 pairs drawn uniformly from the 526,871-protein cohort (seed 42): `query`, `target`, and `dist_<model>`, the Euclidean distance under each of the 15 embedding sets |
| `summary.json` | sampling parameters and convergence checks |
| `convergence.csv` | Spearman ρ between selected models and distance quartiles as the sample grows |

## `new2024.tar`

`new2024/`: a test set of proteins new since the models were trained (from UniRef50 clusters new in
release 2025_01 with no member in any 2024_01 cluster; structures predicted with ColabFold).

| file | contents |
|---|---|
| `new2024.parquet` | 5,466 pairs among 673 proteins, same columns as the pair tables |
| `new2024_exclude_public_before_training.txt` | 5 accessions whose sequence was public before July 2023, the newest training data of any model; the paper drops them and their 4 pairs, leaving 668 proteins and 5,462 pairs |
| `embeddings/<model>.h5` | the 14 pLMs on all 1,225 New2024 candidate proteins (1,173 for ESM1b and CLEAN, which take at most 1022 residues); float32, shape (dim,) for every model here |
