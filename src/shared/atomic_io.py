"""Atomic-write + completeness-guarded skip for pipeline outputs (revision plan v3, B7).

Two failure modes this closes:

1. **Partial final file.** ``save_h5`` (and parquet writers) write incrementally into
   the destination, so a job killed by OOM/walltime leaves a valid-looking but truncated
   *final* file. :func:`atomic_write` writes to a sibling tmp file and ``os.replace``\\s it
   into place — atomic on a POSIX filesystem — so the final path is either the old content
   or the complete new content, never a partial.

2. **"Skip because it exists" consuming a truncated artifact.** :func:`needs_rebuild`
   makes the skip decision on *completeness*, not mere existence, by delegating to the
   fan-in barrier's :func:`~evaluation.analysis_barrier.check_artifact`. A truncated /
   malformed / degenerate artifact is rebuilt instead of silently reused. This shares the
   barrier's single completeness predicate — no second, drift-prone definition of "done".

Overwrite safety: ``mode="timestamp"`` (the **default**) never clobbers an existing
target — new content lands at ``<stem>.<YYYYMMDD_HHMMSS><ext>`` (the CLAUDE.md / B7
convention), leaving the prior artifact for the user to manage; an already-taken
timestamped name is disambiguated with a counter rather than overwritten. ``mode="replace"``
is the explicit opt-in to atomic in-place replacement (wire it to ``--overwrite``).

Threat model: ``os.replace`` makes the swap *atomic* against a killed process (OOM /
walltime / SIGTERM) — the page cache survives process death, so the final file is always
old-or-new, never partial. It does NOT add *durability* against whole-node power loss
(no ``fsync``); that is deliberately out of scope because (a) the audit's hazard is
killed jobs, not power loss, and (b) :func:`needs_rebuild`'s completeness check rebuilds a
corrupt artifact on the next run regardless. Filename convention assumes single-suffix
artifacts (``.parquet`` / ``.h5``); the timestamp is inserted before the final suffix.
"""
from __future__ import annotations

import datetime as _dt
import os
import tempfile
from pathlib import Path
from typing import Callable

from evaluation.analysis_barrier import ArtifactSpec, check_artifact


def _timestamped(path: Path, timestamp: str) -> Path:
    """``dist.parquet`` + ``20260609_154500`` -> ``dist.20260609_154500.parquet``."""
    return path.with_name(f"{path.stem}.{timestamp}{path.suffix}")


def _resolve_target(final_path: Path, mode: str, timestamp: str | None) -> Path:
    if mode == "replace":
        return final_path
    if mode == "timestamp":
        if not final_path.exists():
            return final_path
        ts = timestamp or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        cand = _timestamped(final_path, ts)
        i = 1
        while cand.exists():  # same-second rerun / explicit duplicate stamp
            cand = _timestamped(final_path, f"{ts}_{i}")
            i += 1
        return cand
    raise ValueError(f"unknown mode {mode!r}; expected 'replace' or 'timestamp'")


def atomic_write(
    final_path: Path | str,
    writer: Callable[[Path], None],
    *,
    mode: str = "timestamp",
    timestamp: str | None = None,
) -> Path:
    """Write via a tmp file, then atomically place it; return where it landed.

    Parameters
    ----------
    final_path
        The intended destination.
    writer
        ``writer(tmp_path)`` must write the complete artifact to ``tmp_path``. If it
        raises, the tmp file is removed and the exception propagates — the destination
        is never touched.
    mode
        ``"timestamp"`` (default, fail-safe): if ``final_path`` exists, land at
        ``<stem>.<ts><ext>`` instead of clobbering it (disambiguated if that name is
        also taken). ``"replace"``: explicit opt-in to ``os.replace`` over ``final_path``.
    timestamp
        Explicit ``YYYYMMDD_HHMMSS`` stamp for ``mode="timestamp"`` (defaults to UTC now).

    Returns
    -------
    Path — the path actually written (``final_path`` or the timestamped sibling).
    """
    final_path = Path(final_path)
    target = _resolve_target(final_path, mode, timestamp)
    target.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp gives an OS-guaranteed-unique name in the same dir (so os.replace stays
    # intra-filesystem and atomic), collision-safe across a SLURM array / PID reuse.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f"{target.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        os.replace(tmp, target)  # atomic on the same filesystem
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def needs_rebuild(spec: ArtifactSpec) -> bool:
    """True iff the artifact is absent or fails its completeness contract.

    The completeness-guarded replacement for ``out_path.exists()``: a subcommand should
    rebuild iff this returns True, so a present-but-truncated artifact is not skipped.
    Delegates to the fan-in barrier's :func:`check_artifact` so there is exactly one
    definition of "complete".
    """
    return not check_artifact(spec).ok
