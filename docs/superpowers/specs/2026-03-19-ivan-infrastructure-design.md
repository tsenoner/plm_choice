# Ivan's Infrastructure for pLM Choice Revision

**Date:** 2026-03-19
**Branch:** `feat/ivan-infrastructure`
**Goal:** Build infrastructure for Ivan's three revision tasks, optimize existing code, clean up for Tobias's review.

## 1. GO-Term Semantic Similarity (Wang Method)

### What
A new script that computes Gene Ontology semantic similarity between protein pairs using the Wang (2007) graph-based method. Output is a new target parameter column (`go_wang`) that plugs into the existing training pipeline alongside `fident`, `hfsp`, and `alntmscore`.

### Architecture

**New file:** `src/data_preparation/go_semantic_similarity.py`

**Data inputs:**
- `go-basic.obo` — GO ontology DAG (downloaded from Gene Ontology Consortium, ~35 MB)
- CAFA training annotations — protein-to-GO-term mappings (TSV: protein_id, GO_term, aspect, evidence_code)
- Existing protein pair parquet (the `query`/`target` pairs from the dataset)

**Pipeline:**
1. Download `go-basic.obo` if not cached (from http://purl.obolibrary.org/obo/go/go-basic.obo)
2. Parse GO DAG with `goatools.obo_parser.GODag`
3. Load protein GO annotations (CAFA format or GAF)
4. For each protein pair, compute Wang similarity per GO sub-ontology (MFO, BPO, CCO)
5. Aggregate per-pair: Best-Match Average (BMA) across shared sub-ontology terms
6. Output: add `go_wang_mfo`, `go_wang_bpo`, `go_wang_cco` columns to the parquet

**Key design decisions:**
- Use `goatools` (well-maintained, published in Scientific Reports 2018, has Wang method built-in)
- Pre-compute per-term S-values (Wang's semantic contribution scores) and cache them — these are reusable across all pairs
- Vectorize pair computation: build a protein-to-GO-terms lookup dict, then batch-process pairs
- Support both CAFA TSV and standard GAF annotation formats

**Integration with training pipeline:**
- New target params register in `run_experiments.py` via `--target_params` (already supports arbitrary names)
- Parquet output follows existing convention: `query`, `target`, `go_wang_mfo`, `go_wang_bpo`, `go_wang_cco`
- No changes needed to training code — it reads param columns by name

**Dependencies:** `goatools` (add to pyproject.toml)

### CLI interface
```bash
uv run python src/data_preparation/go_semantic_similarity.py \
    --annotations data/processed/cafa/annotations.tsv \
    --pairs_parquet data/processed/sprot_pre2024/sets/test.parquet \
    --output_parquet data/processed/sprot_pre2024/sets/test_with_go.parquet \
    --obo_path data/reference/go-basic.obo \
    --aspects MFO BPO CCO
```

---

## 2. Randomly Initialized pLM Baseline

### What
Upgrade the baseline from i.i.d. random vectors to proper randomly-initialized model embeddings. A randomly initialized model preserves the architectural prior (attention patterns, layer norms, positional encoding) without any learned biology. This is the correct null hypothesis for "does pretraining help?"

### Architecture

**Modify:** `src/data_preparation/embeddings/embedding_generation.py` — add `--random_init` flag
**Keep:** `src/data_preparation/embeddings/random_embeddings.py` — unchanged (still useful as a simpler baseline)

**Approach:** Add a `--random_init` flag to `embedding_generation.py` that:
1. Loads the model architecture from config (same as normal)
2. Instead of `from_pretrained()`, uses `from_config()` (HuggingFace) which initializes random weights
3. Runs inference normally — the model produces contextual embeddings shaped by architecture, not training
4. Saves to H5 with naming convention `random_init_{model_key}.h5`

**For transformers-based models (ESM2, ProtT5, Ankh, ProstT5):**
```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained(hf_id)
model = model_class(config)  # random weights
```

**For native ESM models (ESM3, ESMC):**
These don't have a clean `from_config()` path. We'll handle this by:
- Extracting the model config from the pretrained model
- Re-initializing all parameters with `model.apply(weight_init_fn)` where `weight_init_fn` resets to standard normal (sigma=0.02, matching ESM's original init)

**Key design decisions:**
- Single flag (`--random_init`) on existing script, not a separate script — reduces code duplication
- Tokenizer is always loaded pretrained (tokenization must be identical to get comparable sequence lengths)
- Random seed is fixed for reproducibility (`--seed` argument, default 42)
- Output naming: `random_init_{model_key}.h5` to distinguish from `random_{dim}.h5` (i.i.d.)

**Comment strategy for existing file:**
- Add header comment block explaining the `--random_init` flag purpose
- Comment at the `from_pretrained` vs `from_config` branch explaining the difference
- Keep all existing code paths untouched

---

## 3. PDB Experimental TM-Score Pipeline

### What
A script that curates a subset of PDB experimental structures and computes pairwise TM-scores as ground truth for structural similarity. This validates whether predicted-structure-based TM-scores (currently used as `alntmscore`) introduce systematic bias.

### Architecture

**New file:** `src/data_preparation/pdb_tmscore.py`

**Pipeline:**
1. **Select PDB subset** — query RCSB Search API for experimental structures matching proteins in the dataset
   - Filter: experimental method in {X-RAY DIFFRACTION, NMR, CRYO-EM}
   - Filter: resolution <= 3.0 A for X-ray
   - Map PDB chains to UniProt IDs (SIFTS mapping from EBI)
   - Intersect with proteins in our dataset
2. **Download structures** — fetch PDB/mmCIF files via RCSB download API
3. **Run TM-align** — pairwise TM-score computation
   - Use TMalign binary (must be on PATH or specified via `--tmalign_path`)
   - Parse TM-score from stdout (normalized by shorter chain)
   - Parallelize with `concurrent.futures.ProcessPoolExecutor`
4. **Output** — parquet with `query`, `target`, `tmscore_exp` column

**Key design decisions:**
- TMalign as external binary (standard in structural bioinformatics, fast C++ implementation)
- SIFTS mapping for UniProt-to-PDB chain correspondence (authoritative source)
- Cache downloaded PDB files in `data/reference/pdb_cache/`
- Process in batches to manage disk space (don't download all PDBs upfront)
- Output format matches existing parquet convention for easy merging

**Dependencies:**
- `requests` (already available via other deps)
- TMalign binary on PATH (external, not pip-installable — document in README)

### CLI interface
```bash
uv run python src/data_preparation/pdb_tmscore.py \
    --pairs_parquet data/processed/sprot_pre2024/sets/test.parquet \
    --output_parquet data/processed/sprot_pre2024/sets/test_with_tmscore_exp.parquet \
    --pdb_cache_dir data/reference/pdb_cache \
    --sifts_mapping data/reference/sifts_uniprot_pdb.tsv \
    --tmalign_path tmalign \
    --max_workers 8 \
    --resolution_cutoff 3.0
```

---

## 4. Distance Computation Optimization

### What
Vectorize the row-by-row Python loop in `distance_computation.py` for a 2-3x speedup. Fix minor inefficiencies.

### Changes to `distance_computation.py`

**4a. Vectorize `_compute_distance_batch()`** (lines 143-176)

Current: iterates rows, does dict lookup + `np.linalg.norm` per pair.

New approach:
```python
def _compute_distance_batch(self, pairs_batch, embedding_name, embeddings):
    queries = pairs_batch["query"].to_list()
    targets = pairs_batch["target"].to_list()

    # Build arrays for all pairs at once
    valid_mask = [(q in embeddings and t in embeddings) for q, t in zip(queries, targets)]

    valid_queries = [q for q, v in zip(queries, valid_mask) if v]
    valid_targets = [t for t, v in zip(targets, valid_mask) if v]

    if valid_queries:
        q_matrix = np.stack([embeddings[q] for q in valid_queries])
        t_matrix = np.stack([embeddings[t] for t in valid_targets])
        valid_dists = np.linalg.norm(q_matrix - t_matrix, axis=1)

    # Fill results with NaN for missing, distances for valid
    distances = np.full(len(pairs_batch), np.nan)
    distances[np.array(valid_mask)] = valid_dists
    return distances.tolist()
```

**4b. Fix `drop_nulls().len()` inefficiency** (line 266)

Replace:
```python
existing_valid = result_df[dist_col].drop_nulls().len()
```
With:
```python
existing_valid = len(result_df) - result_df[dist_col].null_count()
```

**4c. Pre-build embedding matrix for large datasets** (optional optimization)

For the `compute_distances_for_embedding()` method: instead of dict-of-arrays, pre-stack all embeddings into a single numpy matrix with an index mapping. This enables pure numpy operations without per-pair dict lookups.

---

## 5. Code Cleanup

### Comment strategy for modified files
Every pre-existing file we modify gets:
- A comment block at the top of each modified section: `# --- Ivan infrastructure (2026-03-19) ---`
- Inline comments explaining WHY a change was made (not what)
- Original code preserved in comments where behavior changes (not deletions)

### Specific cleanups
- `random_embeddings.py`: Add docstring explaining relationship to new `--random_init` flag
- `distance_computation.py`: Comment the vectorization changes
- `run_experiments.py`: Document the magic numbers (num_workers=10, batch_size=1024, etc.)
- `embedding_generation.py`: Comment the `--random_init` branch

### What we DON'T touch
- Training code (train.py, models.py) — no changes needed
- Evaluation code — works as-is with new params
- Visualization code — works as-is
- Any file not directly related to Ivan's tasks

---

## 6. Dependencies

Add to `pyproject.toml`:
```toml
"goatools>=0.9.9",
```

External binaries (document in README, not pip-managed):
- `tmalign` — TMalign structural alignment (download from Zhang lab)

---

## 7. File Layout (new files only)

```
src/data_preparation/
├── go_semantic_similarity.py    # NEW: Wang method GO similarity
├── pdb_tmscore.py               # NEW: PDB experimental TM-scores
├── distance_computation.py      # MODIFIED: vectorized
└── embeddings/
    ├── embedding_generation.py  # MODIFIED: --random_init flag
    └── random_embeddings.py     # UNCHANGED (add explanatory docstring)
```

---

## 8. Testing Strategy

- Each new script has a `--dry_run` or `--sample_size` flag for quick validation
- GO similarity: validate against known GO-term pairs from literature
- Random init: compare embedding distributions (should differ from i.i.d. random)
- TM-score: validate against known structural pairs
- Distance vectorization: assert numerical equivalence with old code on sample data
