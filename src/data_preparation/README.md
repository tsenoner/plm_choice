# Data Preparation Module

Combines sequence (MMseqs2) and structure (Foldseek) similarity searches with functional similarity scoring (HFSP).

## Pipeline

1. **MMseqs2** → Sequence similarity, filter `fident ≥ 0.3`
2. **Foldseek** → Structure similarity, filter `alntmscore ≥ 0.4`
3. **Merge** → Functional similarity (HFSP) computation, remove self-comparisons

## Quick Start

```bash
# 1. Sequence similarity
./run_mmseqs_all_vs_all.sh sequences.fasta output/mmseqs

# 2. Structure similarity
./run_foldseek_all_vs_all.sh pdb_directory/ output/foldseek

# 3. Merge + functional similarity computation + filtering
python merge_mmseqs_foldseek_datasets.py \
    --mmseqs_file output/mmseqs/results_*/file_all_vs_all.tsv \
    --foldseek_file output/foldseek/results_pdb/pdb_all_vs_all.tsv \
    --output_file merged_dataset_with_hfsp.tsv
```

## Output Dataset

Filtered dataset with high-quality protein pairs:

| Column       | Source   | Description                             |
| ------------ | -------- | --------------------------------------- |
| `query`      | Both     | Query protein identifier                |
| `target`     | Both     | Target protein identifier               |
| `fident`     | MMseqs2  | Sequence identity (≥ 0.3)               |
| `hfsp`       | Computed | Functional similarity (NaN if negative) |
| `alntmscore` | Foldseek | Structural alignment (≥ 0.4)            |

## HFSP Formula

The HFSP (functional similarity score) is computed as:

$$
\begin{align*}
\text{HFSP} &= \text{PIDE} - \begin{cases}
100 & \text{if } L \leq 11 \\
770 \cdot L^{-0.33(1 + e^{L/1000})} & \text{if } 11 < L \leq 450 \\
28.4 & \text{if } L > 450
\end{cases} \\[1em]
\text{PIDE} &= \text{percentage sequence identity of the alignment} \\
L &= \text{ungapped alignment length}
\end{align*}
$$
