"""Adapters that parse the ``cath_labels_319.tsv`` UniProt-style table into the
tidy frames the tested analysis functions expect.

Kept separate from the locked analysis functions so the driver can wrap them
without editing tested code. All parsers are defensive: malformed rows/tokens
are skipped (and the resulting coverage is countable by the caller via ``len``),
never raised — coverage is reported, not assumed (see the recall-FP / EC
coverage discipline).
"""
from __future__ import annotations

import re

import pandas as pd

# Matches a 4-field EC number inside a UniProt "Protein names" string, e.g.
# "chitin synthase (EC 2.4.1.16)" or partial "(EC 3.4.21.-)" / preliminary
# "(EC 1.1.1.n1)". A 2- or 3-field fragment like "(EC 9.9)" deliberately does
# NOT match — it is skipped rather than mis-parsed.
_EC_RE = re.compile(r"\(EC[:\s]\s*(\d+\.\d+\.\d+\.[0-9n-]+)\)")

# A Gene3D / CATH code is exactly four numeric fields C.A.T.H (e.g. 3.90.550.10).
# Topology = first three fields (fold), Homologous-superfamily = all four.
_CATH_RE = re.compile(r"^(\d+\.\d+\.\d+)\.\d+$")


def parse_ec_from_protein_names(
    df: pd.DataFrame,
    id_col: str = "Entry",
    name_col: str = "Protein names",
) -> pd.DataFrame:
    """Extract one EC number per protein from the ``Protein names`` field.

    Parameters
    ----------
    df
        The cath_labels table (needs ``id_col`` + ``name_col`` columns).
    id_col, name_col
        Column names; default to the UniProt export headers.

    Returns
    -------
    pd.DataFrame
        ``[protein_id, ec_number]`` with one row per protein that carries a
        valid 4-field EC number (the first, if several). Proteins with no EC
        are omitted — caller computes coverage as ``len(result) / len(df)``.
    """
    records: list[tuple[str, str]] = []
    for pid, name in zip(df[id_col], df[name_col]):
        if not isinstance(name, str):
            continue
        m = _EC_RE.search(name)
        if m:
            records.append((pid, m.group(1)))
    return pd.DataFrame(records, columns=["protein_id", "ec_number"])


def _parse_gene3d_field(value: object) -> tuple[frozenset[str], frozenset[str]] | None:
    """Parse one ``;``-separated Gene3D cell into (fold, superfamily) sets.

    Returns ``None`` when no valid CATH code is present (caller omits the
    protein). Topology codes (first three fields) populate ``fold``; the full
    four-field codes populate ``superfamily``. Malformed tokens are skipped.
    """
    if not isinstance(value, str):
        return None
    folds: set[str] = set()
    sfams: set[str] = set()
    for token in value.split(";"):
        token = token.strip()
        if not token:
            continue
        m = _CATH_RE.match(token)
        if m:
            folds.add(m.group(1))
            sfams.add(token)
    if not sfams:
        return None
    return frozenset(folds), frozenset(sfams)


def parse_cath_from_gene3d(
    df: pd.DataFrame,
    id_col: str = "Entry",
    gene3d_col: str = "Gene3D",
) -> pd.DataFrame:
    """Parse the ``Gene3D`` column into per-protein CATH label sets.

    Each protein can carry several CATH domains (``;``-separated in the export);
    they are collected as *sets* so the recall-FP positive predicate can score
    multi-domain proteins by set intersection (a target is a positive if it
    shares ANY domain with the query).

    Parameters
    ----------
    df
        The cath_labels table (needs ``id_col`` + ``gene3d_col`` columns).
    id_col, gene3d_col
        Column names; default to the UniProt export headers.

    Returns
    -------
    pd.DataFrame
        ``[protein_id, fold, superfamily, family]`` with one row per protein
        that carries at least one valid Gene3D code. ``fold`` is a frozenset of
        three-field Topology codes; ``superfamily`` a frozenset of four-field
        Homologous-superfamily codes.

        ``family`` is ``None`` — real CATH family labels are an unmet
        people-track input, so this column is a structural placeholder
        (recall_fp's contract requires the column to exist). **Do NOT run
        recall_fp at** ``level="family"`` **until those labels are joined in:**
        the default scalar-equality path would treat ``None == None`` as a match
        for every pair and silently fabricate ``mean_recall_1stFP = 1.0``. The
        driver must scope recall-FP to ``fold``/``superfamily`` (Phase A) until
        the fail-closed guard lands in recall_fp (W1).

        The frozenset columns are meaningful only with the recall-FP
        ``is_positive_fn`` (set-intersection) predicate; the scalar-equality
        default scores exact-set identity and would undercount multi-domain
        positives.

        Proteins with no valid Gene3D code, and rows with a missing/blank
        protein id, are omitted — caller computes coverage as
        ``len(result) / len(df)``. A wholesale-missing ``id_col``/``gene3d_col``
        raises ``KeyError`` (a structural error, distinct from per-row skipping).
    """
    for col in (id_col, gene3d_col):
        if col not in df.columns:
            raise KeyError(
                f"cath_labels table missing required column {col!r}; "
                f"have {list(df.columns)}"
            )
    records: list[tuple[str, frozenset[str], frozenset[str], None]] = []
    for pid, value in zip(df[id_col], df[gene3d_col]):
        if not isinstance(pid, str) or not pid.strip():
            continue
        parsed = _parse_gene3d_field(value)
        if parsed is None:
            continue
        folds, sfams = parsed
        records.append((pid, folds, sfams, None))
    return pd.DataFrame(
        records, columns=["protein_id", "fold", "superfamily", "family"]
    )


def load_cath_labels(
    path,
    id_col: str = "Entry",
    gene3d_col: str = "Gene3D",
) -> pd.DataFrame:
    """Read the cath_labels TSV and return the recall_fp CATH label frame.

    Thin file-IO wrapper over :func:`parse_cath_from_gene3d`. The TSV is the
    UniProt export (``Entry``/``Organism``/``Protein names``/``Gene3D``/...).

    Returns the ``[protein_id, fold, superfamily, family]`` frame; see
    :func:`parse_cath_from_gene3d` for the column semantics and the ``family``
    placeholder caveat.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    return parse_cath_from_gene3d(df, id_col=id_col, gene3d_col=gene3d_col)


def make_cath_is_positive_fn(labels: pd.DataFrame, level: str):
    """Build the recall-FP set-intersection positive predicate for one level.

    The frame produced by :func:`parse_cath_from_gene3d` stores a *set* of CATH
    domains per protein at ``fold``/``superfamily``. Two proteins are positives
    if they share ANY domain at ``level`` — the multi-domain rule recall_fp's
    scalar-equality default cannot express (it would require identical full
    domain sets). Returns a ``(query_id, target_id) -> bool`` closure suitable
    for ``recall_at_first_fp(..., is_positive_fn=...)``.

    Parameters
    ----------
    labels
        The CATH label frame (``protein_id`` + the ``level`` column of frozensets).
    level
        Label column to score on (``"fold"`` or ``"superfamily"``).

    Returns
    -------
    Callable[[str, str], bool]
        Predicate that is ``True`` iff both ids are present with non-null label
        sets that intersect. Unknown ids and null labels (e.g. the ``family``
        placeholder) return ``False`` — the predicate cannot manufacture a
        positive it has no evidence for. A self-match (``q == t``) returns
        ``True`` (a protein shares all its own domains); this is harmless because
        recall_fp excludes the query from its own ranked list.
    """
    if level not in labels.columns:
        raise KeyError(
            f"level {level!r} not in labels columns {list(labels.columns)}"
        )
    lookup = dict(zip(labels["protein_id"], labels[level]))

    def is_positive(query_id: str, target_id: str) -> bool:
        q = lookup.get(query_id)
        t = lookup.get(target_id)
        if q is None or t is None:
            return False
        return len(q & t) > 0

    return is_positive


def _is_wildcard_ec(ec: str) -> bool:
    """True if any of the 4 EC fields is a wildcard ``-`` (a partial / class-only EC)."""
    fields = ec.split(".")
    return any(f == "-" for f in fields) or len(fields) < 4


def _ec_tokens(text: str) -> list[str]:
    """All 4-field EC numbers in a free-text string (uses the shared name regex)."""
    return _EC_RE.findall(text)


def parse_ec(
    df: pd.DataFrame,
    *,
    ec_col: str | None = None,
    id_col: str = "Entry",
    name_col: str = "Protein names",
    wildcard_policy: str = "exclude",
) -> pd.DataFrame:
    """Set-valued EC adapter: one frozenset of EC numbers per protein.

    Prefers a structured ``ec_col`` (a ``;``-separated UniProt "EC number" cell) when
    given; otherwise falls back to the name regex on ``name_col`` (delegating the
    per-string match to the same ``_EC_RE`` the scalar parser uses, so there is one
    regex of record). Returns ``[protein_id, ec_set]`` with one row per protein that
    has >=1 valid EC after policy filtering; proteins with none are omitted (caller
    computes coverage as ``len(result) / len(df)``).

    ``wildcard_policy`` (D8): ``"exclude"`` (default) drops any EC with a ``-`` field
    or fewer than 4 fields (a partial EC has no well-defined hierarchical distance);
    ``"include"`` keeps it verbatim. Preliminary serials (``n1``) are kept either way
    (they have 4 fields). This is additive — the scalar ``parse_ec_from_protein_names``
    is untouched.
    """
    if wildcard_policy not in {"exclude", "include"}:
        raise ValueError(f"wildcard_policy={wildcard_policy!r} must be exclude/include")
    records: list[tuple[str, frozenset]] = []
    for _, row in df.iterrows():
        pid = row[id_col]
        if not isinstance(pid, str) or not pid.strip():
            continue
        if ec_col is not None and isinstance(row.get(ec_col), str):
            raw = [tok.strip() for tok in row[ec_col].split(";") if tok.strip()]
        else:
            name = row.get(name_col)
            raw = _ec_tokens(name) if isinstance(name, str) else []
        kept = {
            ec for ec in raw
            if wildcard_policy == "include" or not _is_wildcard_ec(ec)
        }
        if kept:
            records.append((pid, frozenset(kept)))
    return pd.DataFrame(records, columns=["protein_id", "ec_set"])
