"""Fan-in barrier for the pLM-choice analysis DAG (revision plan v3, B6).

The analysis DAG fans out one array job per pLM, each emitting artifacts
(per-pLM parquet pair tables, per-pLM H5 embeddings). The fan-in steps
(cross-pLM, grid-stat, manifest) must NOT start until every fan-out artifact
exists *and is complete*. This barrier is their `afterok` parent: it validates
each expected artifact and reports a per-artifact verdict, returning a non-zero
exit code if any is missing, unreadable, truncated, malformed, or numerically
degenerate.

Why completeness, not mere existence (B7): an incremental ``h5py.File(path, "w")``
write means a job killed by OOM/walltime leaves a valid-looking but truncated
file the next run would skip as "done". The barrier therefore checks
row/dataset counts, per-vector dimension, key identity, required columns,
finiteness, id-column uniqueness/nullness, and (for embeddings) non-zero norms
— the signatures a partial or scrambled write leaves behind.

Generic by design: the barrier imports no project modules. The caller
(``run_pipeline.py`` / ``submit_analysis_dag.sh``) builds the 15-pLM × metric
grid of :class:`ArtifactSpec` and passes it in, so this module stays trivially
unit-testable with synthetic fixtures.

Exit codes (CLI)
----------------
* ``0`` — every artifact complete.
* ``1`` — at least one artifact missing/incomplete (a *data* failure; downstream
  must not run).
* ``2`` — the barrier could not evaluate (bad ``--spec`` file, malformed JSON,
  missing dependency): an *operator/config* failure, deliberately distinct from
  a data failure so SLURM logs disambiguate the two.

CLI::

    python -m evaluation.analysis_barrier --spec barrier_spec.json

where ``barrier_spec.json`` is ``{"artifacts": [ {<ArtifactSpec fields>}, ... ]}``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


class SpecError(Exception):
    """The barrier spec itself is malformed (operator error → exit 2)."""


# ── specs / status ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ArtifactSpec:
    """One expected fan-out artifact and the completeness contract it must meet.

    A 0-row / 0-dataset artifact always fails: an empty fan-out output is never
    a valid result (it is the canonical "job created the file then died" case).
    """

    label: str
    path: Path
    expected_rows: int | None = None
    # parquet pair-table contracts
    required_columns: tuple[str, ...] = ()
    finite_columns: tuple[str, ...] = ()
    unique_columns: tuple[str, ...] = ()
    non_null_columns: tuple[str, ...] = ()
    # h5 embedding contracts
    expected_dim: int | None = None
    expected_keys: tuple[str, ...] | None = None
    require_positive_norm: bool = False
    min_norm: float = 0.0
    kind: str = "auto"  # "parquet" | "h5" | "auto" (infer from suffix)

    def resolved_kind(self) -> str:
        if self.kind != "auto":
            return self.kind
        suffix = Path(self.path).suffix.lower()
        if suffix in (".h5", ".hdf5"):
            return "h5"
        if suffix in (".parquet", ".pq"):
            return "parquet"
        raise ValueError(
            f"cannot infer kind from suffix {suffix!r}; set kind= explicitly"
        )


@dataclass(frozen=True)
class ArtifactStatus:
    label: str
    path: Path
    ok: bool
    n_rows: int | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BarrierReport:
    statuses: tuple[ArtifactStatus, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.statuses)

    @property
    def failures(self) -> tuple[ArtifactStatus, ...]:
        return tuple(s for s in self.statuses if not s.ok)

    def format_report(self) -> str:
        # ASCII only: SLURM logs under LANG=C raise UnicodeEncodeError on
        # fancy glyphs, which on the failure path would mask the report itself.
        lines = []
        for s in self.statuses:
            tag = "PASS" if s.ok else "FAIL"
            rows = "" if s.n_rows is None else f" (rows={s.n_rows})"
            lines.append(f"[{tag}] {s.label}{rows} - {s.path}")
            for r in s.reasons:
                lines.append(f"         x {r}")
        n_fail = len(self.failures)
        n_total = len(self.statuses)
        summary = (
            f"barrier OK: {n_total}/{n_total} artifacts complete"
            if n_fail == 0
            else f"barrier FAILED: {n_fail}/{n_total} artifact(s) incomplete"
        )
        lines.append(summary)
        return "\n".join(lines)


# ── per-kind checkers ─────────────────────────────────────────────────────────
def _check_parquet(spec: ArtifactSpec) -> tuple[int | None, list[str]]:
    # Imports outside the try so a missing dependency surfaces as an ImportError
    # (operator/env fault → exit 2), never as a bogus "this artifact is corrupt".
    import numpy as np
    import pandas as pd

    reasons: list[str] = []
    try:
        df = pd.read_parquet(spec.path)
    except Exception as e:  # noqa: BLE001 — any read failure = incomplete artifact
        return None, [f"unreadable: {type(e).__name__}: {e}"]

    n_rows = int(len(df))
    if n_rows == 0:
        reasons.append("artifact is empty (0 rows)")
    if spec.expected_rows is not None and n_rows != spec.expected_rows:
        reasons.append(f"row count {n_rows} != expected {spec.expected_rows}")

    missing = [c for c in spec.required_columns if c not in df.columns]
    if missing:
        reasons.append(f"missing columns: {sorted(missing)}")

    for col in spec.finite_columns:
        if col not in df.columns:
            reasons.append(f"finite-check column absent: {col}")
            continue
        vals = df[col].to_numpy()
        if not np.issubdtype(vals.dtype, np.number):
            reasons.append(f"finite-check column not numeric: {col} ({vals.dtype})")
            continue
        n_bad = int((~np.isfinite(vals)).sum())
        if n_bad:
            reasons.append(f"non-finite values in column {col}: {n_bad}")

    for col in spec.non_null_columns:
        if col not in df.columns:
            reasons.append(f"non-null-check column absent: {col}")
            continue
        n_null = int(df[col].isna().sum())
        if n_null:
            reasons.append(f"null values in column {col}: {n_null}")

    if spec.unique_columns:
        present = [c for c in spec.unique_columns if c in df.columns]
        absent = [c for c in spec.unique_columns if c not in df.columns]
        if absent:
            reasons.append(f"uniqueness-check columns absent: {sorted(absent)}")
        if present:
            n_dup = int(df.duplicated(subset=present).sum())
            if n_dup:
                reasons.append(
                    f"duplicate rows on {tuple(present)}: {n_dup}"
                )

    return n_rows, reasons


def _check_h5(spec: ArtifactSpec) -> tuple[int | None, list[str]]:
    import h5py
    import numpy as np

    reasons: list[str] = []
    try:
        f = h5py.File(spec.path, "r")
    except Exception as e:  # noqa: BLE001
        return None, [f"unreadable: {type(e).__name__}: {e}"]

    with f:
        keys = list(f.keys())
        n_rows = len(keys)
        if n_rows == 0:
            reasons.append("artifact is empty (0 datasets)")
        if spec.expected_rows is not None and n_rows != spec.expected_rows:
            reasons.append(
                f"dataset count {n_rows} != expected {spec.expected_rows}"
            )

        if spec.expected_keys is not None:
            have = set(keys)
            want = set(spec.expected_keys)
            n_missing = len(want - have)
            n_extra = len(have - want)
            if n_missing or n_extra:
                reasons.append(
                    f"key set mismatch: {n_missing} missing, {n_extra} unexpected"
                )

        need_scan = (
            spec.require_positive_norm or spec.expected_dim is not None
        )
        if need_scan:
            n_nonfinite = 0
            n_zero_norm = 0
            n_bad_dim = 0
            n_nondataset = 0
            n_unreadable = 0
            for k in keys:
                node = f[k]
                if not isinstance(node, h5py.Dataset):
                    n_nondataset += 1
                    continue
                try:
                    vec = np.asarray(node[:])  # native dtype (no float64 upcast)
                except Exception:  # noqa: BLE001 — isolate a single bad dataset
                    n_unreadable += 1
                    continue
                if spec.expected_dim is not None and (
                    vec.ndim != 1 or vec.shape[-1] != spec.expected_dim
                ):
                    n_bad_dim += 1
                if spec.require_positive_norm:
                    if not np.all(np.isfinite(vec)):
                        n_nonfinite += 1
                        continue
                    if float(np.linalg.norm(vec)) <= spec.min_norm:
                        n_zero_norm += 1
            if n_nondataset:
                reasons.append(f"non-Dataset members (groups?): {n_nondataset}")
            if n_unreadable:
                reasons.append(f"unreadable datasets: {n_unreadable}")
            if n_bad_dim:
                reasons.append(
                    f"unexpected embedding dim: {n_bad_dim} vector(s) "
                    f"!= {spec.expected_dim}"
                )
            if n_nonfinite:
                reasons.append(f"non-finite (NaN/Inf) embeddings: {n_nonfinite}")
            if n_zero_norm:
                thresh = "zero" if spec.min_norm == 0.0 else f"<= {spec.min_norm}"
                reasons.append(f"{thresh}-norm embeddings: {n_zero_norm}")

    return n_rows, reasons


# ── public API ────────────────────────────────────────────────────────────────
def check_artifact(spec: ArtifactSpec) -> ArtifactStatus:
    """Validate a single artifact against its completeness contract.

    Never raises for an artifact-level problem — including an un-inferrable
    ``kind`` — so one bad entry cannot abort a whole barrier run. Only an
    environment fault (e.g. a missing pandas/h5py) propagates.
    """
    path = Path(spec.path)
    if not path.exists():
        return ArtifactStatus(
            label=spec.label, path=path, ok=False, n_rows=None,
            reasons=(f"missing: {path}",),
        )

    try:
        kind = spec.resolved_kind()
    except ValueError as e:
        return ArtifactStatus(
            label=spec.label, path=path, ok=False, n_rows=None,
            reasons=(f"unknown kind: {e}",),
        )

    if kind == "parquet":
        n_rows, reasons = _check_parquet(spec)
    elif kind == "h5":
        n_rows, reasons = _check_h5(spec)
    else:
        n_rows, reasons = None, [f"unknown kind: {kind}"]

    return ArtifactStatus(
        label=spec.label, path=path, ok=(len(reasons) == 0),
        n_rows=n_rows, reasons=tuple(reasons),
    )


def run_barrier(specs: Iterable[ArtifactSpec]) -> BarrierReport:
    """Validate every artifact; the report is OK iff all pass."""
    return BarrierReport(statuses=tuple(check_artifact(s) for s in specs))


def _spec_from_dict(d: dict, idx: int) -> ArtifactSpec:
    if not isinstance(d, dict):
        raise SpecError(f"artifact #{idx}: expected an object, got {type(d).__name__}")
    for required in ("label", "path"):
        if required not in d:
            raise SpecError(f"artifact #{idx}: missing required key {required!r}")
    keys = d.get("expected_keys")
    return ArtifactSpec(
        label=d["label"],
        path=Path(d["path"]),
        expected_rows=d.get("expected_rows"),
        required_columns=tuple(d.get("required_columns", ())),
        finite_columns=tuple(d.get("finite_columns", ())),
        unique_columns=tuple(d.get("unique_columns", ())),
        non_null_columns=tuple(d.get("non_null_columns", ())),
        expected_dim=d.get("expected_dim"),
        expected_keys=tuple(keys) if keys is not None else None,
        require_positive_norm=bool(d.get("require_positive_norm", False)),
        min_norm=float(d.get("min_norm", 0.0)),
        kind=d.get("kind", "auto"),
    )


def _load_specs(spec_path: str | Path) -> list[ArtifactSpec]:
    path = Path(spec_path)
    try:
        text = path.read_text()
    except FileNotFoundError as e:
        raise SpecError(f"spec file not found: {path}") from e
    except OSError as e:
        raise SpecError(f"spec file unreadable: {path}: {e}") from e
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise SpecError(f"spec file is not valid JSON: {path}: {e}") from e
    if not isinstance(payload, dict) or "artifacts" not in payload:
        raise SpecError(f"spec file must be an object with an 'artifacts' list: {path}")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list):
        raise SpecError(f"'artifacts' must be a list: {path}")
    return [_spec_from_dict(d, i) for i, d in enumerate(artifacts)]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="analysis_barrier",
        description="Fan-in barrier: fail unless every expected artifact is complete.",
    )
    ap.add_argument(
        "--spec", required=True,
        help="JSON file: {'artifacts': [ {label, path, expected_rows, ...}, ... ]}",
    )
    args = ap.parse_args(argv)

    try:
        specs = _load_specs(args.spec)
    except SpecError as e:
        print(f"barrier: CONFIG ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    report = run_barrier(specs)
    # OK summary on stdout; the failure report on stderr so it lands in the
    # SLURM .err the operator reads when a barrier-gated job is held.
    stream = sys.stdout if report.ok else sys.stderr
    print(report.format_report(), file=stream, flush=True)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
