#!/usr/bin/env python3
"""
GO-Term Semantic Similarity (Wang Method)

Computes Gene Ontology semantic similarity between protein pairs using the
Wang (2007) graph-based method. Outputs a new target parameter column that
plugs into the existing pLM Choice training pipeline alongside fident, hfsp,
and alntmscore.

The Wang method computes similarity based on the topology of the GO DAG:
each term's "semantic value" is the sum of contributions from all its
ancestors, weighted by edge type (is_a vs part_of). Protein-pair similarity
is computed via Best-Match Average (BMA) across shared GO sub-ontology terms.

References:
    Wang JZ et al. (2007) "A new method to measure the semantic similarity
    of GO terms." Bioinformatics 23(10):1274-81.

Usage:
    uv run python src/data_preparation/go_semantic_similarity.py \
        --annotations data/processed/cafa/annotations.tsv \
        --pairs_parquet data/processed/sprot_pre2024/sets/test.parquet \
        --output_parquet data/processed/sprot_pre2024/sets/test_with_go.parquet \
        --obo_path data/reference/go-basic.obo

Created: 2026-03-19 (Ivan infrastructure for pLM Choice revision)
"""

import argparse
import logging
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# GO sub-ontology namespaces
GO_ASPECTS = {
    "molecular_function": "MFO",
    "biological_process": "BPO",
    "cellular_component": "CCO",
    # Also accept short forms from annotation files
    "F": "MFO",
    "P": "BPO",
    "C": "CCO",
    "MFO": "MFO",
    "BPO": "BPO",
    "CCO": "CCO",
}

# Wang method edge weights (standard values from the paper)
EDGE_WEIGHTS = {
    "is_a": 0.8,
    "part_of": 0.6,
}


# --------------------------------------------------------------------------- #
#                            GO DAG HANDLING
# --------------------------------------------------------------------------- #


class GOTerm:
    """Minimal GO term representation for Wang method computation."""

    __slots__ = ("id", "name", "namespace", "parents", "is_obsolete")

    def __init__(self, id: str, name: str = "", namespace: str = ""):
        self.id = id
        self.name = name
        self.namespace = namespace
        self.parents: List[Tuple[str, str]] = []  # [(parent_id, relation_type), ...]
        self.is_obsolete = False


def download_obo(obo_path: Path) -> None:
    """Download go-basic.obo from Gene Ontology Consortium if not cached."""
    url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    logger.info(f"Downloading GO ontology from {url} ...")
    urllib.request.urlretrieve(url, str(obo_path))
    logger.info(f"Saved to {obo_path} ({obo_path.stat().st_size / 1e6:.1f} MB)")


def parse_obo(obo_path: Path) -> Dict[str, GOTerm]:
    """
    Parse go-basic.obo into a dict of GOTerm objects.

    Lightweight parser that extracts only what the Wang method needs:
    term IDs, namespaces, and parent relationships (is_a + part_of).
    We avoid goatools dependency to keep this self-contained.
    """
    terms: Dict[str, GOTerm] = {}
    current_term: Optional[GOTerm] = None
    in_term_block = False

    with open(obo_path) as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                in_term_block = True
                current_term = GOTerm(id="")
                continue
            elif line.startswith("[") and line.endswith("]"):
                # End of a [Term] block, entering [Typedef] or similar
                if in_term_block and current_term and current_term.id:
                    if not current_term.is_obsolete:
                        terms[current_term.id] = current_term
                in_term_block = False
                current_term = None
                continue

            if not in_term_block or current_term is None:
                continue

            if line.startswith("id: "):
                current_term.id = line[4:]
            elif line.startswith("name: "):
                current_term.name = line[6:]
            elif line.startswith("namespace: "):
                current_term.namespace = line[11:]
            elif line.startswith("is_a: "):
                # Format: "is_a: GO:0008150 ! biological_process"
                parent_id = line[6:].split(" !")[0].strip()
                current_term.parents.append((parent_id, "is_a"))
            elif line.startswith("relationship: part_of "):
                parent_id = line[22:].split(" !")[0].strip()
                current_term.parents.append((parent_id, "part_of"))
            elif line.startswith("is_obsolete: true"):
                current_term.is_obsolete = True

    # Don't forget the last term if file doesn't end with another block
    if in_term_block and current_term and current_term.id and not current_term.is_obsolete:
        terms[current_term.id] = current_term

    logger.info(
        f"Parsed {len(terms)} GO terms from {obo_path.name} "
        f"(MFO: {sum(1 for t in terms.values() if t.namespace == 'molecular_function')}, "
        f"BPO: {sum(1 for t in terms.values() if t.namespace == 'biological_process')}, "
        f"CCO: {sum(1 for t in terms.values() if t.namespace == 'cellular_component')})"
    )
    return terms


# --------------------------------------------------------------------------- #
#                        WANG METHOD IMPLEMENTATION
# --------------------------------------------------------------------------- #


class WangSimilarity:
    """
    Computes GO semantic similarity using the Wang (2007) method.

    For a given GO term A, the S-value of each ancestor t is defined
    recursively:
        S_A(A) = 1
        S_A(t) = max over children c of t: { w_e * S_A(c) }
    where w_e is the edge weight (0.8 for is_a, 0.6 for part_of).

    The semantic value of A is SV(A) = sum of S_A(t) for all ancestors t.

    Similarity between terms A and B:
        sim(A, B) = sum of (S_A(t) + S_B(t)) for t in ancestors(A) ∩ ancestors(B)
                    / (SV(A) + SV(B))
    """

    def __init__(self, go_terms: Dict[str, GOTerm]):
        self.go_terms = go_terms
        # Cache: go_id -> {ancestor_id: s_value}
        self._s_value_cache: Dict[str, Dict[str, float]] = {}

    def _compute_s_values(self, term_id: str) -> Dict[str, float]:
        """
        Compute S-values for a term and all its ancestors (recursive with cache).

        Returns dict mapping ancestor_id -> S_A(ancestor_id).
        """
        if term_id in self._s_value_cache:
            return self._s_value_cache[term_id]

        if term_id not in self.go_terms:
            self._s_value_cache[term_id] = {term_id: 1.0}
            return self._s_value_cache[term_id]

        # S_A(A) = 1
        s_values: Dict[str, float] = {term_id: 1.0}

        # BFS/DFS up the DAG, propagating weighted contributions
        stack = [(term_id, 1.0)]
        while stack:
            current_id, current_s = stack.pop()
            term = self.go_terms.get(current_id)
            if term is None:
                continue

            for parent_id, relation in term.parents:
                weight = EDGE_WEIGHTS.get(relation, 0.4)  # default weight for unknown
                propagated_s = current_s * weight

                # Keep max contribution if multiple paths reach same ancestor
                if parent_id not in s_values or propagated_s > s_values[parent_id]:
                    s_values[parent_id] = propagated_s
                    stack.append((parent_id, propagated_s))

        self._s_value_cache[term_id] = s_values
        return s_values

    def term_similarity(self, term_a: str, term_b: str) -> float:
        """Compute Wang similarity between two GO terms."""
        s_a = self._compute_s_values(term_a)
        s_b = self._compute_s_values(term_b)

        sv_a = sum(s_a.values())
        sv_b = sum(s_b.values())

        if sv_a + sv_b == 0:
            return 0.0

        # Sum contributions from shared ancestors
        shared_ancestors = set(s_a.keys()) & set(s_b.keys())
        numerator = sum(s_a[t] + s_b[t] for t in shared_ancestors)

        return numerator / (sv_a + sv_b)

    def protein_similarity_bma(
        self,
        terms_a: Set[str],
        terms_b: Set[str],
    ) -> float:
        """
        Compute Best-Match Average (BMA) similarity between two sets of GO terms.

        BMA = (sum of max_sim for each term in A against all of B
             + sum of max_sim for each term in B against all of A)
             / (|A| + |B|)
        """
        if not terms_a or not terms_b:
            return np.nan

        # Forward: for each term in A, find best match in B
        forward_sum = 0.0
        for ta in terms_a:
            best = max(self.term_similarity(ta, tb) for tb in terms_b)
            forward_sum += best

        # Backward: for each term in B, find best match in A
        backward_sum = 0.0
        for tb in terms_b:
            best = max(self.term_similarity(ta, tb) for ta in terms_a)
            backward_sum += best

        return (forward_sum + backward_sum) / (len(terms_a) + len(terms_b))


# --------------------------------------------------------------------------- #
#                        ANNOTATION LOADING
# --------------------------------------------------------------------------- #


def load_annotations_tsv(
    annotations_path: Path,
    go_terms: Dict[str, GOTerm],
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Load protein-to-GO-term annotations from a TSV file.

    Expected TSV columns (tab-separated, no header or with header):
        protein_id  GO_term  aspect
    where aspect is one of: F/P/C or MFO/BPO/CCO or full namespace.

    Also supports GAF format (21 columns, '!' comment lines).

    Returns:
        Dict mapping protein_id -> {"MFO": {GO:xxxx, ...}, "BPO": {...}, "CCO": {...}}
    """
    annotations: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: {"MFO": set(), "BPO": set(), "CCO": set()}
    )

    skipped_terms = 0
    loaded_terms = 0

    with open(annotations_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue

            parts = line.split("\t")

            # Auto-detect format
            if len(parts) >= 15:
                # GAF format: col 1 = protein_id (DB_Object_ID), col 4 = GO_id, col 8 = aspect
                protein_id = parts[1]  # DB_Object_ID
                go_id = parts[4]
                aspect_raw = parts[8]
            elif len(parts) >= 3:
                # Simple TSV: protein_id, GO_term, aspect
                protein_id = parts[0]
                go_id = parts[1]
                aspect_raw = parts[2]
            else:
                continue

            # Normalize aspect
            aspect = GO_ASPECTS.get(aspect_raw)
            if aspect is None:
                continue

            # Only keep terms that exist in the ontology
            if go_id in go_terms:
                annotations[protein_id][aspect].add(go_id)
                loaded_terms += 1
            else:
                skipped_terms += 1

    logger.info(
        f"Loaded {loaded_terms} annotations for {len(annotations)} proteins "
        f"(skipped {skipped_terms} terms not in ontology)"
    )
    return dict(annotations)


# --------------------------------------------------------------------------- #
#                        PAIR SIMILARITY COMPUTATION
# --------------------------------------------------------------------------- #


def compute_pair_similarities(
    pairs_df: pl.DataFrame,
    annotations: Dict[str, Dict[str, Set[str]]],
    wang: WangSimilarity,
    aspects: List[str],
    batch_size: int = 1000,
) -> Dict[str, List[float]]:
    """
    Compute GO Wang similarity for all protein pairs.

    Args:
        pairs_df: DataFrame with 'query' and 'target' columns
        annotations: protein_id -> {aspect: {GO terms}}
        wang: WangSimilarity instance
        aspects: which GO aspects to compute (e.g. ["MFO", "BPO", "CCO"])
        batch_size: pairs to process before logging progress

    Returns:
        Dict mapping column name -> list of similarity values
    """
    results: Dict[str, List[float]] = {f"go_wang_{a.lower()}": [] for a in aspects}

    queries = pairs_df["query"].to_list()
    targets = pairs_df["target"].to_list()

    annotated_count = 0
    total = len(queries)

    for i in tqdm(range(total), desc="Computing GO Wang similarity", unit="pair"):
        q_id = queries[i]
        t_id = targets[i]

        q_annot = annotations.get(q_id, {})
        t_annot = annotations.get(t_id, {})

        has_any = False
        for aspect in aspects:
            col_name = f"go_wang_{aspect.lower()}"
            q_terms = q_annot.get(aspect, set())
            t_terms = t_annot.get(aspect, set())

            if q_terms and t_terms:
                sim = wang.protein_similarity_bma(q_terms, t_terms)
                results[col_name].append(float(sim))
                has_any = True
            else:
                results[col_name].append(np.nan)

        if has_any:
            annotated_count += 1

    logger.info(
        f"Computed similarities for {annotated_count}/{total} pairs "
        f"({annotated_count / total * 100:.1f}% had annotations in both proteins)"
    )
    return results


# --------------------------------------------------------------------------- #
#                                MAIN
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Compute GO-term semantic similarity (Wang method) between protein pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to GO annotations file (TSV: protein_id, GO_term, aspect; or GAF format)",
    )
    parser.add_argument(
        "--pairs_parquet",
        type=Path,
        required=True,
        help="Path to Parquet file with protein pairs (must have 'query' and 'target' columns)",
    )
    parser.add_argument(
        "--output_parquet",
        type=Path,
        required=True,
        help="Path for output Parquet file with GO similarity columns added",
    )
    parser.add_argument(
        "--obo_path",
        type=Path,
        default=None,
        help="Path to go-basic.obo file. If not provided or missing, will download.",
    )
    parser.add_argument(
        "--aspects",
        nargs="+",
        default=["MFO", "BPO", "CCO"],
        choices=["MFO", "BPO", "CCO"],
        help="GO sub-ontologies to compute",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Limit number of pairs to process (for testing)",
    )

    args = parser.parse_args()

    # --- Validate inputs ---
    if not args.annotations.exists():
        logger.error(f"Annotations file not found: {args.annotations}")
        sys.exit(1)
    if not args.pairs_parquet.exists():
        logger.error(f"Pairs parquet not found: {args.pairs_parquet}")
        sys.exit(1)

    # --- Load or download GO ontology ---
    obo_path = args.obo_path
    if obo_path is None:
        obo_path = args.annotations.parent / "go-basic.obo"
    if not obo_path.exists():
        obo_path.parent.mkdir(parents=True, exist_ok=True)
        download_obo(obo_path)

    go_terms = parse_obo(obo_path)

    # --- Load annotations ---
    annotations = load_annotations_tsv(args.annotations, go_terms)

    # --- Load protein pairs ---
    pairs_df = pl.read_parquet(args.pairs_parquet)
    logger.info(f"Loaded {len(pairs_df)} protein pairs from {args.pairs_parquet}")

    if args.sample_size:
        pairs_df = pairs_df.head(args.sample_size)
        logger.info(f"Sampling {args.sample_size} pairs for testing")

    # Check annotation coverage
    all_proteins = set(pairs_df["query"].unique().to_list()) | set(
        pairs_df["target"].unique().to_list()
    )
    annotated_proteins = set(annotations.keys()) & all_proteins
    logger.info(
        f"Annotation coverage: {len(annotated_proteins)}/{len(all_proteins)} proteins "
        f"({len(annotated_proteins) / len(all_proteins) * 100:.1f}%)"
    )

    # --- Compute similarities ---
    wang = WangSimilarity(go_terms)
    similarity_columns = compute_pair_similarities(
        pairs_df, annotations, wang, args.aspects
    )

    # --- Merge results ---
    result_df = pairs_df.clone()
    for col_name, values in similarity_columns.items():
        result_df = result_df.with_columns(
            pl.Series(name=col_name, values=values)
        )

    # --- Save ---
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(args.output_parquet)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("GO SEMANTIC SIMILARITY COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output_parquet}")
    for col_name in similarity_columns:
        series = result_df[col_name]
        valid = len(series) - series.null_count()
        if valid > 0:
            logger.info(
                f"  {col_name}: {valid}/{len(series)} valid "
                f"({valid / len(series) * 100:.1f}%), "
                f"mean={series.mean():.3f}, std={series.std():.3f}"
            )
        else:
            logger.info(f"  {col_name}: no valid values")

    logger.info(f"\nS-value cache size: {len(wang._s_value_cache)} terms")


if __name__ == "__main__":
    main()
