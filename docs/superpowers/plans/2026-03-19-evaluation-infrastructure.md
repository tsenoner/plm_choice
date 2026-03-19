# Evaluation & Validation Infrastructure Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build four remaining infrastructure scripts: EC-number hierarchy distances, BRENDA/HFSP validation, recall-at-first-false-positive metric, and AUROC at SCOP classification levels.

**Architecture:** Each task produces either a new data-preparation script (following the parquet I/O pattern from `go_semantic_similarity.py`) or extends `metrics.py` with new evaluation functions. All output integrates into the existing training/evaluation pipeline.

**Tech Stack:** Python 3.12, polars, numpy, scipy, sklearn, requests. No new pip dependencies beyond what's already in pyproject.toml.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/data_preparation/ec_hierarchy_distance.py` | Create | EC-number distance between protein pairs |
| `src/data_preparation/brenda_hfsp_validation.py` | Create | BRENDA annotation fetch + HFSP validation |
| `src/evaluation/retrieval_metrics.py` | Create | Recall@first-FP, AUROC at SCOP levels |
| `src/evaluation/metrics.py` | Unchanged | Existing regression metrics (not modified) |
| `docs/todo.md` | Modify | Update task list |
| `tests/test_ec_hierarchy.py` | Create | Unit tests for EC distance logic |
| `tests/test_retrieval_metrics.py` | Create | Unit tests for new evaluation metrics |

**Design decision:** New evaluation metrics go in a separate `retrieval_metrics.py` rather than modifying `metrics.py`. Reason: `metrics.py` is regression-only and is imported by `evaluate.py`/`evaluate_multiple.py` — adding classification imports there would bloat those paths. The new file is focused on retrieval/classification evaluation.

---

### Task 1: EC-Number Hierarchy Distance

EC numbers have 4 levels (e.g. `3.4.21.9`). Distance = level at which two EC numbers first differ. This is a clean ordinal metric for functional similarity.

**Files:**
- Create: `src/data_preparation/ec_hierarchy_distance.py`
- Create: `tests/test_ec_hierarchy.py`

- [ ] **Step 1: Write tests for EC distance logic**

```python
# tests/test_ec_hierarchy.py
import pytest
from src.data_preparation.ec_hierarchy_distance import ec_distance, parse_ec_number

def test_parse_ec_number():
    assert parse_ec_number("3.4.21.9") == (3, 4, 21, 9)
    assert parse_ec_number("3.4.21.-") == (3, 4, 21, None)
    assert parse_ec_number("3.-.-.-") == (3, None, None, None)

def test_ec_distance_identical():
    assert ec_distance("3.4.21.9", "3.4.21.9") == 0

def test_ec_distance_level4():
    # Same up to level 3, differ at level 4
    assert ec_distance("3.4.21.9", "3.4.21.4") == 1

def test_ec_distance_level3():
    # Same up to level 2, differ at level 3
    assert ec_distance("3.4.21.9", "3.4.24.9") == 2

def test_ec_distance_level1():
    # Differ at first level
    assert ec_distance("1.1.1.1", "3.4.21.9") == 4

def test_ec_distance_with_wildcards():
    # Wildcard (-) means unknown — distance is NaN
    assert ec_distance("3.4.21.-", "3.4.21.9") is None

def test_ec_distance_partial():
    # Incomplete EC numbers
    assert ec_distance("3.4.-.-", "3.4.21.9") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `VIRTUAL_ENV= uv run pytest tests/test_ec_hierarchy.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement EC distance script**

Create `src/data_preparation/ec_hierarchy_distance.py` following the same pattern as `go_semantic_similarity.py`:
- `parse_ec_number(ec_str)` → tuple of 4 ints (None for wildcards)
- `ec_distance(ec_a, ec_b)` → int 0-4 or None if either has wildcards
- `load_ec_annotations(path)` → dict mapping protein_id → set of EC numbers
- Supports UniProt DAT, TSV, and ID-mapping formats
- CLI: `--annotations`, `--pairs_parquet`, `--output_parquet`
- Output columns: `ec_distance_min` (minimum EC distance across all EC pairs for a protein pair), `ec_distance_max`, `ec_distance_mean`
- Downloads EC annotations from UniProt if not cached: `https://rest.uniprot.org/uniprotkb/stream?query=*&fields=accession,ec&format=tsv`

- [ ] **Step 4: Run tests to verify they pass**

Run: `VIRTUAL_ENV= uv run pytest tests/test_ec_hierarchy.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_preparation/ec_hierarchy_distance.py tests/test_ec_hierarchy.py
git commit -m "feat: add EC-number hierarchy distance computation"
```

---

### Task 2: BRENDA/HFSP Validation

Validate HFSP scores against curated BRENDA functional annotations for beta-lactamases. The idea: beta-lactamases have well-characterized functional classes (Class A/B/C/D). Pairs within the same class should have high HFSP, pairs across classes should have low HFSP.

**Files:**
- Create: `src/data_preparation/brenda_hfsp_validation.py`

- [ ] **Step 1: Implement BRENDA validation script**

Create `src/data_preparation/brenda_hfsp_validation.py`:
- Fetch beta-lactamase entries from UniProt (EC 3.5.2.6) with class annotations
- Classify into Ambler classes (A/B/C/D) from protein family annotations
- For pairs in the dataset that are beta-lactamases, compute:
  - Within-class HFSP distribution (should be high)
  - Between-class HFSP distribution (should be low)
  - Separation statistics (Mann-Whitney U, Cohen's d)
- Output: validation report (JSON + optional matplotlib figure)
- CLI: `--pairs_parquet` (must already have `hfsp` column), `--output_dir`, `--enzyme_ec 3.5.2.6`
- Generalize: accept any EC number, not just beta-lactamases, for future validation

- [ ] **Step 2: Syntax check**

Run: `python3 -c "import py_compile; py_compile.compile('src/data_preparation/brenda_hfsp_validation.py', doraise=True)"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/data_preparation/brenda_hfsp_validation.py
git commit -m "feat: add BRENDA/HFSP validation for enzyme functional classes"
```

---

### Task 3: Recall-at-First-False-Positive

A retrieval metric: rank protein pairs by predicted similarity (embedding distance), scan the ranking, count how many true similar pairs you find before the first false positive.

**Files:**
- Create: `src/evaluation/retrieval_metrics.py`
- Create: `tests/test_retrieval_metrics.py`

- [ ] **Step 1: Write tests for retrieval metrics**

```python
# tests/test_retrieval_metrics.py
import numpy as np
import pytest
from src.evaluation.retrieval_metrics import recall_at_first_fp, auroc_at_level

def test_recall_at_first_fp_perfect():
    # All positives ranked first
    distances = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels = np.array([True, True, True, False, False])
    result = recall_at_first_fp(distances, labels)
    assert result["recall_at_first_fp"] == 1.0
    assert result["n_retrieved"] == 3

def test_recall_at_first_fp_immediate_failure():
    # First pair is a false positive
    distances = np.array([0.1, 0.2, 0.3])
    labels = np.array([False, True, True])
    result = recall_at_first_fp(distances, labels)
    assert result["recall_at_first_fp"] == 0.0
    assert result["n_retrieved"] == 0

def test_recall_at_first_fp_partial():
    # Some TPs before first FP
    distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = np.array([True, True, False, True, False])
    result = recall_at_first_fp(distances, labels)
    assert result["recall_at_first_fp"] == pytest.approx(2/3)
    assert result["n_retrieved"] == 2

def test_auroc_at_level_perfect():
    # Perfect separation
    distances = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([True, True, False, False])
    assert auroc_at_level(distances, labels) == pytest.approx(1.0)

def test_auroc_at_level_random():
    # Random — should be ~0.5
    np.random.seed(42)
    distances = np.random.rand(1000)
    labels = np.random.choice([True, False], 1000)
    auc = auroc_at_level(distances, labels)
    assert 0.4 < auc < 0.6  # roughly 0.5

def test_auroc_at_level_inverted():
    # Inverted (high distance = similar) — should still work with ascending=True
    distances = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([True, True, False, False])
    auc = auroc_at_level(distances, labels, lower_is_similar=False)
    assert auc == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `VIRTUAL_ENV= uv run pytest tests/test_retrieval_metrics.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement retrieval metrics**

Create `src/evaluation/retrieval_metrics.py`:

```python
"""
Retrieval and classification evaluation metrics for pLM Choice.

Supplements the regression metrics in metrics.py with:
- recall_at_first_fp: retrieval quality before first false positive
- auroc_at_level: AUROC for binary classification at SCOP/ECOD levels
- evaluate_retrieval: end-to-end evaluation combining all retrieval metrics

These metrics support the paper's reframing from regression to
classification/retrieval evaluation.
"""
```

Functions:
- `recall_at_first_fp(distances, labels, lower_is_similar=True)` → dict with recall, n_retrieved, n_positives
- `auroc_at_level(distances, labels, lower_is_similar=True)` → float AUROC
- `evaluate_retrieval(pairs_df, distance_col, classification_col, levels)` → dict of metrics per level

- [ ] **Step 4: Run tests to verify they pass**

Run: `VIRTUAL_ENV= uv run pytest tests/test_retrieval_metrics.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/retrieval_metrics.py tests/test_retrieval_metrics.py
git commit -m "feat: add recall-at-first-FP and AUROC retrieval metrics"
```

---

### Task 4: AUROC at Fold/Superfamily/Family Level

Extend the retrieval metrics with a CLI script that computes AUROC at each SCOP/ECOD hierarchy level. Uses the functions from Task 3.

**Files:**
- Create: `src/evaluation/classification_eval.py`

- [ ] **Step 1: Implement classification evaluation script**

Create `src/evaluation/classification_eval.py`:
- Load protein pairs parquet with distance columns + structural classification
- For each distance column (one per pLM embedding):
  - At each level (Family, Superfamily, Fold):
    - Binary label: "same level?" → True/False
    - Compute AUROC using `auroc_at_level()` from Task 3
    - Compute recall-at-first-FP using `recall_at_first_fp()` from Task 3
- Output: summary table (parquet + optional CSV) with columns: embedding, level, auroc, recall_at_first_fp, n_positives, n_negatives
- Supports SCOP (sf_id, fa_id, fold_id columns), ECOD (T/H/X/F group columns), or generic hierarchy columns
- CLI: `--pairs_parquet`, `--distance_columns`, `--classification_source {scop,ecod,custom}`, `--output_dir`

- [ ] **Step 2: Syntax check**

Run: `python3 -c "import py_compile; py_compile.compile('src/evaluation/classification_eval.py', doraise=True)"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/classification_eval.py
git commit -m "feat: add AUROC/recall classification evaluation at SCOP/ECOD levels"
```

---

### Task 5: Update docs and final commit

**Files:**
- Modify: `docs/todo.md`

- [ ] **Step 1: Update todo.md with new infrastructure**

Add section documenting the 4 new scripts and their CLI usage.

- [ ] **Step 2: Commit**

```bash
git add docs/todo.md
git commit -m "docs: update todo with evaluation infrastructure"
```
