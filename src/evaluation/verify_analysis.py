"""Freeze-integrity gate for the analysis DAG (revision plan v3, Phase 0 item 2).

``verify_analysis`` is the hard ``afterok`` precondition the plan requires before any figure
consumes an artifact (Gate C). It is distinct from two neighbours:

* the **fan-in barrier** (``analysis_barrier``) checks that each fan-out *artifact* is present
  and complete (row counts, finiteness, norms) — it answers "did every job finish cleanly?";
* the upstream ``verify_manifest`` is presence-only — "do the files exist?".

``verify_analysis`` answers a different question: **is the analysis still anchored to the
frozen canonical set?** Concretely it asserts:

1. the canonical FASTA on disk still hashes to the frozen ``canonical_content_sha256`` — the
   sequence set has not drifted out from under results already computed against it;
2. the NEW-3 ``esm1b_paired_policy`` is *locked* (non-null) whenever the freeze carries an
   esm1b coverage block — so a paired stat never runs against an undecided cohort policy;
3. (optional) a named analysis input's population matches the frozen id set via
   :func:`~evaluation.population.assert_population`, with esm1b permitted to be its capped
   267-subset.

It reuses the freeze manifest (:mod:`evaluation.canonical_set`) and
:func:`~evaluation.population.assert_population` so there is exactly one definition of "the
canonical set" — no drift-prone second copy.

Exit codes (CLI), matching :mod:`evaluation.analysis_barrier`:

* ``0`` — every check passed.
* ``1`` — an integrity check failed (drift / unlocked policy / population mismatch): a *data*
  failure; downstream figures must not run.
* ``2`` — the gate could not evaluate (manifest missing/malformed, an unparseable ``--fasta``,
  a missing dependency): an *operator/config* failure, deliberately distinct so SLURM logs
  disambiguate it from a data failure. (A FASTA that parses but whose content *drifted* is a
  data failure → exit 1 via the hash mismatch — that is the gate's primary signal.)

CLI::

    python -m evaluation.verify_analysis --manifest freeze/canonical_set_319.json --fasta <fasta>
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from evaluation.canonical_set import canonical_content_sha256, parse_fasta
from evaluation.population import PopulationError, assert_population

_REQUIRED_MANIFEST_KEYS = ("schema_version", "canonical_content_sha256", "ids", "n_proteins")


class ManifestError(ValueError):
    """The freeze manifest is malformed or internally inconsistent (operator error → exit 2).

    Subclasses ``ValueError`` so callers (and tests) that catch ``ValueError`` still work,
    while the name documents the intent at the raise/except sites.
    """


# ── report ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class VerifyReport:
    checks: tuple[Check, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        # A gate that ran zero checks is NOT ok: a false-green is worse than a false-red.
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.ok)

    def format_report(self) -> str:
        # ASCII only (SLURM LANG=C); mirror analysis_barrier's report style.
        lines = []
        for c in self.checks:
            tag = "PASS" if c.ok else "FAIL"
            lines.append(f"[{tag}] {c.name}")
            for r in c.reasons:
                lines.append(f"         x {r}")
        n_fail = len(self.failures)
        n_total = len(self.checks)
        lines.append(
            f"verify_analysis OK: {n_total}/{n_total} checks passed"
            if n_fail == 0
            else f"verify_analysis FAILED: {n_fail}/{n_total} check(s) failed"
        )
        return "\n".join(lines)


# ── manifest loading ──────────────────────────────────────────────────────────────
def load_manifest(path: Path | str) -> dict:
    """Load + validate a freeze manifest's structure and internal consistency.

    Raises ``FileNotFoundError`` if absent, ``ManifestError`` (a ``ValueError``) if the JSON
    is malformed, a required key is missing, a key is wrong-typed, or the manifest is
    internally inconsistent (``len(ids) != n_proteins`` / ``n_pairs != C(n, 2)``). All of
    these are operator/config faults the CLI surfaces as exit 2 — so a structurally broken
    manifest never reaches the integrity checks and crashes them with a stray AttributeError.
    """
    path = Path(path)
    text = path.read_text()  # FileNotFoundError propagates (operator error)
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as e:
        raise ManifestError(f"manifest is not valid JSON: {path}: {e}") from e
    if not isinstance(manifest, dict):
        raise ManifestError(f"manifest must be a JSON object: {path}")
    missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ManifestError(f"manifest missing required key(s): {missing} ({path})")

    # Type guards: a wrong-typed key must fail as a clean config error here, not as an
    # uncaught AttributeError/TypeError deep inside a check.
    ids = manifest["ids"]
    if not isinstance(ids, list):
        raise ManifestError(f"manifest 'ids' must be a list, got {type(ids).__name__} ({path})")
    n_proteins = manifest["n_proteins"]
    if not isinstance(n_proteins, int):
        raise ManifestError(
            f"manifest 'n_proteins' must be an int, got {type(n_proteins).__name__} ({path})"
        )
    esm1b = manifest.get("esm1b")
    if esm1b is not None and not isinstance(esm1b, dict):
        raise ManifestError(
            f"manifest 'esm1b' must be an object or null, got {type(esm1b).__name__} ({path})"
        )

    # Internal consistency: the gate exists to catch drift, so a self-inconsistent freeze
    # (e.g. a hand-edited manifest) must fail rather than pass a partial check.
    if len(ids) != n_proteins:
        raise ManifestError(
            f"manifest inconsistent: len(ids)={len(ids)} != n_proteins={n_proteins} ({path})"
        )
    n_pairs = manifest.get("n_pairs")
    if n_pairs is not None and n_pairs != n_proteins * (n_proteins - 1) // 2:
        raise ManifestError(
            f"manifest inconsistent: n_pairs={n_pairs} != C({n_proteins}, 2)="
            f"{n_proteins * (n_proteins - 1) // 2} ({path})"
        )
    return manifest


# ── individual checks (each returns a list of reasons; [] == passed) ──────────────
def verify_fasta_unchanged(manifest: dict, fasta_path: Path | str) -> list[str]:
    """The FASTA on disk must still hash to the frozen ``canonical_content_sha256``."""
    observed = canonical_content_sha256(parse_fasta(fasta_path))
    frozen = manifest["canonical_content_sha256"]
    if observed != frozen:
        return [
            f"canonical content sha256 mismatch: on-disk {observed[:12]}... "
            f"!= frozen {frozen[:12]}... (the sequence set has drifted)"
        ]
    return []


def verify_policy_locked(manifest: dict) -> list[str]:
    """If the freeze carries an esm1b block, ``esm1b_paired_policy`` must be locked (non-null).

    Gates on *presence* (``isinstance dict``), not truthiness: an empty ``esm1b: {}`` block is
    "coverage tracked but policy absent" — exactly the undecided state to catch — so it must
    flag, not silently pass via a falsy-dict short-circuit.
    """
    esm1b = manifest.get("esm1b")
    if not isinstance(esm1b, dict):
        return []  # no esm1b block (None); a wrong-typed one was rejected by load_manifest
    if esm1b.get("esm1b_paired_policy") is None:
        return [
            "esm1b_paired_policy is null (NEW-3 not locked); refusing to gate a paired "
            "analysis against an undecided esm1b cohort policy."
        ]
    return []


def verify_population(
    manifest: dict, observed_keys, *, name: str, capped: bool = False
) -> list[str]:
    """An analysis input's population must match the frozen id set (esm1b may be capped)."""
    try:
        assert_population(observed_keys, manifest["ids"], name=name, allow_capped=capped)
    except PopulationError as e:
        return [str(e)]
    return []


# ── orchestration ─────────────────────────────────────────────────────────────────
def verify_analysis(
    manifest_path: Path | str,
    *,
    fasta_path: Path | str | None = None,
    population_inputs: dict[str, "Iterable[str]"] | None = None,
    esm1b_name: str = "esm1b",
) -> VerifyReport:
    """Run the freeze-integrity gate and return a :class:`VerifyReport`.

    Always checks the esm1b policy lock. If ``fasta_path`` is given, also checks the FASTA is
    unchanged (the anti-drift control). If ``population_inputs`` (a ``{pLM_name: observed_ids}``
    map) is given, asserts each cohort against the frozen set in one invocation — realizing
    S3's "assert the population before every analysis" at the gate rather than by convention.
    The pLM named ``esm1b_name`` is allowed to be its capped subset iff the freeze records an
    esm1b coverage block (NEW-3 ``footnote_esm1b_out``).
    """
    manifest = load_manifest(manifest_path)
    checks: list[Check] = []

    if fasta_path is not None:
        reasons = verify_fasta_unchanged(manifest, fasta_path)
        checks.append(Check("fasta_unchanged", not reasons, tuple(reasons)))

    reasons = verify_policy_locked(manifest)
    checks.append(Check("esm1b_policy_locked", not reasons, tuple(reasons)))

    if population_inputs is not None:
        esm1b_capped = isinstance(manifest.get("esm1b"), dict)
        for name in sorted(population_inputs):
            capped = esm1b_capped and name == esm1b_name
            reasons = verify_population(
                manifest, population_inputs[name], name=name, capped=capped
            )
            checks.append(Check(f"population:{name}", not reasons, tuple(reasons)))

    return VerifyReport(checks=tuple(checks))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="verify_analysis",
        description="Freeze-integrity gate: fail unless the analysis is still anchored to "
        "the frozen canonical set.",
    )
    ap.add_argument("--manifest", required=True, help="Path to the freeze manifest JSON.")
    ap.add_argument(
        "--fasta",
        required=True,
        help="Canonical FASTA to re-hash against the frozen content sha256 (the anti-drift "
        "control — required so a CLI run can never green-light without checking drift).",
    )
    args = ap.parse_args(argv)

    try:
        report = verify_analysis(args.manifest, fasta_path=args.fasta)
    except (FileNotFoundError, ValueError, ManifestError) as e:
        print(f"verify_analysis: CONFIG ERROR: {e}", file=sys.stderr, flush=True)
        return 2

    stream = sys.stdout if report.ok else sys.stderr
    print(report.format_report(), file=stream, flush=True)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
