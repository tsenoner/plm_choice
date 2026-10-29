# Which pLM to choose? Embeddings, protein pairs and analysis data

Data for the manuscript *Which pLM to choose?* by Tobias Senoner, Ivan Koludarov, Joshua Günther,
Amarda Shehu, Burkhard Rost and Yana Bromberg.

- Preprint: <https://doi.org/10.1101/2025.10.30.685515>
- Code: <https://github.com/tsenoner/plm_choice>
- Cite all versions: concept DOI [10.5281/zenodo.17469267](https://doi.org/10.5281/zenodo.17469267), which
  resolves to the newest version. This version: 10.5281/zenodo.22935411.

The study compares 14 protein language models (pLMs) and a random Gaussian control on 75,849,972
Swiss-Prot protein pairs, each scored for sequence identity (PIDE), HFSP and structural similarity
(TM-score). It separates the signal already in raw embedding distances from the signal a supervised
probe extracts, and compares embeddings with an MMseqs2 homology search for transferring EC numbers
and GO molecular-function terms. Formats are described in `DATA_DICTIONARY.md`.

## Files

| file | size | contents |
|---|---:|---|
| `README.md`, `DATA_DICTIONARY.md`, `CITATION.cff` | | this description, the file formats, citation metadata |
| `ankh_base.h5` … `random_1024.h5` (15 files) | 32.7 GB | one embedding per protein for each of the 14 pLMs and the random Gaussian control; 540,881 Swiss-Prot proteins (526,871 for ESM1b and CLEAN) |
| `pair_tables_splits.tar` | 1.0 GB | the deduplicated pair set: train, validation and test pairs with PIDE, HFSP and TM-score |
| `pair_table_counts.tar` | 10 kB | pair counts per split and target |
| `identical_sequence_mask.tar` | 2.7 MB | which pairs join two identical sequences |
| `probes_full100_p10.tar` | 3.1 GB | the probe grid: trained probes and raw-distance scores, 15 embedding sets × 3 targets |
| `probes_pca_cohort.tar` | 3.1 GB | the same grid on 128-dimensional PCA projections |
| `probes_randinit_full100.tar` | 2.3 GB | the same grid on 11 untrained (random-initialisation) networks |
| `functional_annotation.tar` | 202 MB | EC and GO-MF cohorts, 1-nearest-neighbour annotation-transfer results, the MMseqs2 homology baseline |
| `experimental_tm.tar` | 0.8 MB | test pairs rescored from experimental PDB structures with US-align |
| `distance_histograms.tar` | 59 MB | per-model distance distributions over random pairs and over aligner-found pairs |
| `random_pairs.tar` | 601 MB | 5,000,000 uniformly drawn protein pairs with their embedding distances under every model |
| `new2024.tar` | 74 MB | the New2024 test set: pairs among proteins new in UniRef50 2025_01, the 5 excluded proteins, embeddings |

Total about 43 GB. The `.tar` archives are uncompressed; unpack with `tar -xf <file>`.

## Data in brief

- **Proteins.** Swiss-Prot release 2024_01, 542,238 proteins, clustered at 30% identity and split by cluster
  70/15/15 into training, validation and test. The evaluation cohort is the 526,871 proteins present in all
  15 embedding files, i.e. those of at most 1022 residues (the context window of ESM1b, inherited by CLEAN).
  The pair tables hold all pairs; the cohort filter is applied when pairs are loaded and removes 940,833
  (1.2%).
- **Embeddings.** The mean over per-residue final-layer representations. ProtT5, ProtTucker and the random
  Gaussian control are stored as float16, the other twelve as float32, so distances recomputed from these
  files match the paper's to within about 2 × 10⁻⁴ relative.
- **Pairs.** From an all-against-all MMseqs2 search (PIDE, HFSP) and a Foldseek search of AlphaFold DB v4
  models (TM-score), filtered for reliability and deduplicated: each protein pair appears once. 75,074,822
  pairs carry a TM-score and 36,349,349 carry PIDE and HFSP.
- **Probes.** A feed-forward network (FNN) that compresses each of the two embeddings to 64 dimensions with
  one shared layer and concatenates them, trained on 100% of the training pairs with early stopping
  (patience 10) and scored by Spearman ρ on the test split. Each grid also holds the
  raw-distance read-out (Euclidean distance, no fitted parameters) for the same cells.

## Quick start

```python
import h5py, numpy as np, pandas as pd

pairs = pd.read_parquet("sprot_pre2024_e1/sets/test.parquet")       # from pair_tables_splits.tar
row = pairs.iloc[0]
with h5py.File("esm2_650m.h5") as h:
    a = np.asarray(h[row["query"]], dtype=np.float32).ravel()
    b = np.asarray(h[row["target"]], dtype=np.float32).ravel()
print(np.linalg.norm(a - b), row[["fident", "hfsp", "alntmscore"]].to_dict())
```

`.ravel()` matters for `prott5.h5` and `prottucker.h5`, which store each vector with shape (1, d).

## Changes from version 1 (2025)

- The embedding files are replaced in place by the ones behind the revised results; they cover a different
  set of proteins (for example, `esm2_3b.h5` grew from 4.6 to 5.7 GB).
- The pair set is deduplicated: 75.8 million pairs, each once (version 1: 113 million rows).
- The probes are retrained on all training pairs instead of a 10% sample, with patience 10 instead of 5,
  and come with a PCA grid and an untrained-network control.
- Added: the identical-sequence mask, functional annotation transfer, experimental-structure TM-scores,
  distance histograms, the random pairs and the New2024 test set.
- Removed: `supplementary_data.zip`, version 1's pair sets, probes and figures, all superseded here. It
  remains in version 1, [10.5281/zenodo.17469268](https://doi.org/10.5281/zenodo.17469268).

## Licence

CC BY 4.0. The embeddings were computed with the published weights of each model; see each model's own
licence for its use.
