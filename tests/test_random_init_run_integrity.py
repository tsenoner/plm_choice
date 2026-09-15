"""The random-init baseline must not be able to fabricate its own error bar.

D-6 reports the untrained arm as mean±sd over seeds 0/1/2. Two defects made that
number unreliable in ways that produce a clean exit code:

* the default output filename carried no seed, so all three seeds resolved to the
  same ``random_init_<model>.h5``;
* the writer opens HDF5 in append mode and its "already exists" branch
  *increments the success counter* before continuing, so seeds 1 and 2 would skip
  every protein, report a full success and exit 0.

Together they mean three runs produce one file of seed-0 vectors, and the paper
reports ``sd = 0.000`` — a fabricated error bar, in a resubmission whose whole
problem is trust in the numbers. These tests pin both halves shut.

The pooling tests cover the batched extraction path: batching is worth 25-75x on
this workload, but a plain ``.mean()`` over a padded batch silently averages in
padding, and the shorter the protein the more wrong it gets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data_preparation.embeddings.embedding_generation import (  # noqa: E402
    MODEL_CONFIGS,
    _reinit_weights,
    default_output_path,
    effective_max_seq_len,
    load_model_and_tokenizer,
    process_sequences_and_save,
)
from tests.conftest import tiny_esm_config  # noqa: E402

# --------------------------------------------------------------------------- #
#            the random arm must run at its twin's compute precision
# --------------------------------------------------------------------------- #


def _stub_config(monkeypatch, **extra):
    """A MODEL_CONFIGS entry whose model loads from a local tiny config."""
    pytest.importorskip("transformers")
    from transformers import EsmModel

    import data_preparation.embeddings.embedding_generation as eg

    monkeypatch.setattr(
        eg.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: tiny_esm_config()),
    )
    cfg = {
        "hf_id": "stub/esm",
        "loader": "transformers",
        "model_class": EsmModel,
        "tokenizer_class": None,
        "family_key": "esm_transformer",
    }
    cfg.update(extra)
    return cfg


def test_post_load_hook_runs_on_the_random_arm(monkeypatch):
    """`pretrained - random_init` is only a statement about weights.

    ProtT5/ProstT5 carry a `.half()` post-load hook, so skipping it for the
    random arm means the untrained model computes in fp32 while its pretrained
    twin computed in fp16. That difference lands directly in the headline
    difference and is indistinguishable from a pretraining effect.

    Drives the real loader: asserting that a predicate returns True proves
    nothing about whether ``load_model_and_tokenizer`` ever calls the hook.
    """
    fired = []
    cfg = _stub_config(monkeypatch, post_load_hook=lambda m: fired.append(m) or m)

    load_model_and_tokenizer(
        "stub", cfg, None, torch.device("cpu"), random_init=True, random_seed=0
    )

    assert fired, "the random arm skipped the post-load hook"


def test_post_load_hook_runs_on_the_pretrained_arm_too(monkeypatch):
    fired = []
    cfg = _stub_config(monkeypatch, post_load_hook=lambda m: fired.append(m) or m)
    from transformers import EsmModel

    monkeypatch.setattr(
        EsmModel,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: EsmModel(tiny_esm_config())),
    )

    load_model_and_tokenizer(
        "stub", cfg, None, torch.device("cpu"), random_init=False, random_seed=0
    )

    assert fired, "the pretrained arm skipped the post-load hook"


def test_random_init_honours_a_dtype_declared_only_in_load_kwargs(monkeypatch):
    """from_pretrained applies torch_dtype; constructing from a config does not.

    A model that expresses fp16 through ``load_kwargs`` alone would otherwise give
    its untrained twin fp32 — the precision confound, reintroduced silently.
    """
    cfg = _stub_config(monkeypatch, load_kwargs={"torch_dtype": torch.float16})

    model, _, _ = load_model_and_tokenizer(
        "stub", cfg, None, torch.device("cpu"), random_init=True, random_seed=0
    )

    assert all(p.dtype == torch.float16 for p in model.parameters())


# --------------------------------------------------------------------------- #
#                 different seeds must mean different weights
# --------------------------------------------------------------------------- #


def test_transformers_random_init_is_seeded_and_reproducible(monkeypatch):
    """The defect this whole branch exists to prevent, at its root.

    ``sd = 0.000`` does not need a filename collision to happen: if the seed
    stopped reaching the weights — say ``torch.manual_seed`` drifted below the
    constructor — all three seeds would produce identical models and every
    filename test here would still pass. 10 of the 13 grid arms take this
    ``transformers`` path, so it is the one that matters most.
    """
    cfg = _stub_config(monkeypatch)

    def weights(seed):
        model, _, _ = load_model_and_tokenizer(
            "stub", cfg, None, torch.device("cpu"), random_init=True, random_seed=seed
        )
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    a0, a0_again, a1 = weights(0), weights(0), weights(1)

    assert torch.equal(a0, a0_again), "same seed must reproduce the same weights"
    assert not torch.equal(a0, a1), (
        "seeds 0 and 1 produced identical weights — the untrained arm's error bar "
        "would be exactly 0.000 regardless of how many seeds were run"
    )


# --------------------------------------------------------------------------- #
#                  esm1b needs an untrained twin like the rest
# --------------------------------------------------------------------------- #


def test_esm1b_is_configured():
    """Without this entry the random-init baseline covers 12 of 16 paper arms."""
    assert "esm1b" in MODEL_CONFIGS
    cfg = MODEL_CONFIGS["esm1b"]
    assert cfg["hf_id"] == "facebook/esm1b_t33_650M_UR50S"
    assert cfg["family_key"] == "esm_transformer"
    assert cfg["loader"] == "transformers"


def test_esm1b_position_limit_leaves_room_for_the_position_id_offset():
    """The cap is 1022 residues, and the two missing slots are easy to lose.

    ``max_position_embeddings`` is 1026, but HuggingFace derives position ids as
    ``cumsum(mask) + padding_idx`` with ``padding_idx = 1``, so T tokens reach id
    T+1 and the table admits at most T = 1024 tokens — <cls> + 1022 residues +
    <eos>. Verified against the real checkpoint: 1022 passes, 1023 and 1024 both
    raise IndexError.

    Pinning 1024 here would not have been a harmless off-by-two: on CUDA the
    out-of-range embedding lookup is a device-side assert that kills every
    subsequent batch in the task, and bucketing runs shortest-first, so the run
    loses its whole long tail and still exits 0.
    """
    cfg = MODEL_CONFIGS["esm1b"]
    max_position_embeddings, padding_idx, n_special = 1026, 1, 2
    assert cfg["max_positions"] == (
        max_position_embeddings - padding_idx - 1 - n_special
    ) == 1022


@pytest.mark.parametrize("model_key", sorted(MODEL_CONFIGS))
def test_position_limit_is_declared_only_where_the_architecture_has_one(model_key):
    """State the contract, rather than pinning one other model's absence of a key.

    Rotary and native-ESM models have no absolute-position ceiling; a model that
    declares one must have it honoured when the caller passes no --max_seq_len.
    """
    cfg = MODEL_CONFIGS[model_key]
    ceiling = cfg.get("max_positions")
    if ceiling is None:
        assert effective_max_seq_len(cfg, None) is None
    else:
        assert effective_max_seq_len(cfg, None) == ceiling
        # Asking for more than the architecture supports is clamped, not honoured.
        assert effective_max_seq_len(cfg, ceiling + 1000) == ceiling
        # The caller may always ask for less.
        assert effective_max_seq_len(cfg, ceiling - 1) == ceiling - 1


# --------------------------------------------------------------------------- #
#                     seeds must not collide on one filename
# --------------------------------------------------------------------------- #


def test_random_init_output_path_carries_the_seed():
    """Without this, seeds 0/1/2 all resolve to the same file."""
    fasta = Path("/data/sprot.fasta")
    paths = {
        default_output_path(fasta, "esm2_650m", random_init=True, random_seed=s)
        for s in (0, 1, 2)
    }
    assert len(paths) == 3, f"seeds collapsed onto {len(paths)} path(s): {paths}"


def test_random_init_output_path_names_the_seed_explicitly():
    got = default_output_path(
        Path("/data/sprot.fasta"), "esm2_650m", random_init=True, random_seed=1
    )
    assert got.name == "random_init_esm2_650m_seed1.h5"


def test_pretrained_output_path_is_unchanged():
    """The pretrained naming is load-bearing for existing data — do not disturb it."""
    got = default_output_path(
        Path("/data/sprot.fasta"), "esm2_650m", random_init=False, random_seed=0
    )
    assert got.name == "sprot_esm2_650m.h5"


def test_output_path_sanitises_slashes_in_model_key():
    got = default_output_path(
        Path("/data/sprot.fasta"), "facebook/esm2", random_init=True, random_seed=2
    )
    assert "/" not in got.name
    assert got.name == "random_init_facebook_esm2_seed2.h5"


# --------------------------------------------------------------------------- #
#                  a random-init run must refuse to append
# --------------------------------------------------------------------------- #


def test_random_init_refuses_to_append_into_an_existing_file(tmp_path):
    """Belt-and-braces against the skip-and-count-as-success branch.

    Even with per-seed filenames, a re-run must not quietly resume into a file
    written by a different seed. It has to fail loudly instead.
    """
    existing = tmp_path / "random_init_esm2_650m_seed0.h5"
    h5py = pytest.importorskip("h5py")
    with h5py.File(existing, "w") as fh:
        fh.create_dataset("P12345", data=np.zeros(4, dtype=np.float32))

    with pytest.raises(FileExistsError):
        process_sequences_and_save(
            sequences_to_process=[("P99999", "MKT")],
            model=None,
            tokenizer=None,
            family_key="esm_transformer",
            embedding_type="per_protein",
            device=torch.device("cpu"),
            h5_output_path=existing,
            max_seq_len=None,
            model_key_for_filename="esm2_650m",
            random_init=True,
        )


def test_pretrained_run_may_still_resume(tmp_path, read_h5):
    """Resuming a long pretrained run is a feature and must keep working.

    Uses the ``read_h5`` fixture rather than ``h5py.File(..., "r")`` directly —
    see its docstring for why a same-process read-back needs a collection first.
    """
    h5py = pytest.importorskip("h5py")
    existing = tmp_path / "sprot_esm2_650m.h5"
    stored = np.arange(4, dtype=np.float32)
    with h5py.File(existing, "w") as fh:
        fh.create_dataset("P12345", data=stored)

    # Feeding back an id the file already holds exercises the resume path: with
    # "w-" this would raise, and with "w" the dataset would be gone.
    n_done, n_failed = process_sequences_and_save(
        sequences_to_process=[("P12345", "MKT")],
        model=None,
        tokenizer=None,
        family_key="esm_transformer",
        embedding_type="per_protein",
        device=torch.device("cpu"),
        h5_output_path=existing,
        max_seq_len=None,
        model_key_for_filename="esm2_650m",
        random_init=False,
    )

    assert n_done == 1, "resume must find the stored embedding, not recompute it"
    assert n_failed == 0, "an already-present protein is not a failure"

    fh = read_h5(existing)
    assert "P12345" in fh, "resume must not truncate the file"
    assert np.array_equal(fh["P12345"][:], stored), (
        "resume must leave the existing embedding byte-for-byte intact"
    )


# --------------------------------------------------------------------------- #
#              re-init must survive LayerNorms that have no bias
# --------------------------------------------------------------------------- #


def test_reinit_handles_layernorm_without_bias():
    """ESM-3 / ESM-C build LayerNorms with bias=False; zeros_(None) raises.

    The Linear branch already guards on `bias is not None`; the LayerNorm branch
    does not. Without this the three ESM-C/ESM-3 arms — 3 of the top 4 on fident
    and hfsp — cannot be random-init'd at all.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.LayerNorm(8, bias=False),
    )
    _reinit_weights(model, seed=0)

    ln = model[1]
    assert ln.bias is None
    assert torch.allclose(ln.weight, torch.ones_like(ln.weight))


def test_reinit_still_zeroes_layernorm_bias_when_present():
    model = torch.nn.Sequential(torch.nn.LayerNorm(8))
    with torch.no_grad():
        model[0].bias.fill_(0.5)
    _reinit_weights(model, seed=0)
    assert torch.allclose(model[0].bias, torch.zeros_like(model[0].bias))
    assert torch.allclose(model[0].weight, torch.ones_like(model[0].weight))


# --------------------------------------------------------------------------- #
#            an incomplete run must not look like a finished one
# --------------------------------------------------------------------------- #
#
# (Pooling itself is covered in test_batched_embedding_extraction.py, which owns
# that contract. What belongs here is what the RUN reports back.)


def _failing_model(*_args, **_kwargs):
    raise RuntimeError("simulated device-side assert")


@pytest.mark.parametrize("token_budget", [0, 64], ids=["unbatched", "batched"])
def test_a_run_that_drops_proteins_reports_them_as_failures(tmp_path, token_budget):
    """Both paths must agree on what "processed" means.

    Every failure inside the writer is caught and logged so one bad protein does
    not lose the run, which makes the returned counts the only signal that the
    file is incomplete. They previously disagreed: the batched path counted only
    new writes and the unbatched path also counted skips, so the same cohort
    reported two different totals depending on --token_budget.
    """
    pytest.importorskip("h5py")
    pytest.importorskip("transformers")
    from tests.test_batched_driver_equivalence import StubTokenizer

    n_ok, n_failed = process_sequences_and_save(
        sequences_to_process=[("P1", "MKTA"), ("P2", "MKVAA")],
        model=_failing_model,
        tokenizer=StubTokenizer(),
        family_key="esm_transformer",
        embedding_type="per_protein",
        device=torch.device("cpu"),
        h5_output_path=tmp_path / "out.h5",
        max_seq_len=None,
        model_key_for_filename="stub",
        token_budget=token_budget,
    )

    assert n_ok == 0
    assert n_failed == 2, (
        "a run that embedded nothing must not be indistinguishable from a "
        "complete one — main() turns this into a non-zero exit"
    )


def test_batched_and_unbatched_agree_on_a_resumed_cohort(tmp_path):
    """The already-present count must not depend on --token_budget."""
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("transformers")
    from tests.test_batched_driver_equivalence import StubTokenizer, tiny_esm

    model = tiny_esm()
    counts = []
    for i, token_budget in enumerate([0, 64]):
        path = tmp_path / f"resume_{i}.h5"
        with h5py.File(path, "w") as fh:
            fh.create_dataset("P1", data=np.zeros(32, dtype=np.float32))
        counts.append(
            process_sequences_and_save(
                sequences_to_process=[("P1", "MKTA"), ("P2", "MKVAA")],
                model=model,
                tokenizer=StubTokenizer(),
                family_key="esm_transformer",
                embedding_type="per_protein",
                device=torch.device("cpu"),
                h5_output_path=path,
                max_seq_len=None,
                model_key_for_filename="stub",
                token_budget=token_budget,
            )
        )

    assert counts[0] == counts[1] == (2, 0)


# --------------------------------------------------------------------------- #
#  The shared-cohort guard
#
#  A 12-task array reads ONE sprot.fasta. pyfaidx rewrites the ``.fai`` index
#  non-atomically (plain ``open('w')`` + ``copyfileobj``, guarded only by a
#  *threading* lock, which is nothing across processes), and the run used to
#  delete that shared index on its way out — so every task that finished forced
#  every later task to rebuild it.
#
#  A reader landing mid-rewrite does not raise. It gets a silent SUBSET, or —
#  when the cut falls inside the trailing lenc/lenb field, leaving five still
#  parseable columns — real accessions carrying the WRONG RESIDUES. Both write a
#  plausible HDF5 and exit 0, and the num_failed contract cannot see either,
#  because the missing proteins were never enumerated in the first place.
# --------------------------------------------------------------------------- #

import subprocess  # noqa: E402
import sys  # noqa: E402

_MODULE = "data_preparation.embeddings.embedding_generation"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke the module the sbatch invokes, so exit codes are the real ones."""
    return subprocess.run(
        [sys.executable, "-m", _MODULE, *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )


def _write_fasta(path: Path, n: int) -> None:
    path.write_text("".join(f">P{i:05d}\nMKV{'A' * (10 + i)}\n" for i in range(n)))


def test_cohort_guard_accepts_the_expected_count(tmp_path):
    fasta = tmp_path / "c.fasta"
    _write_fasta(fasta, 5)
    r = _run_cli(str(fasta), "esm2_8m", "--random_init", "--expect_sequences", "5",
                 "--output_hdf5_file", str(tmp_path / "o.h5"))
    assert "cohort size mismatch" not in r.stderr
    assert r.returncode != 8


def test_cohort_guard_rejects_a_truncated_index(tmp_path):
    """A .fai cut at a record boundary yields a subset with no exception."""
    fasta = tmp_path / "c.fasta"
    _write_fasta(fasta, 5)
    pyfaidx = pytest.importorskip("pyfaidx")
    pyfaidx.Fasta(str(fasta))
    fai = Path(str(fasta) + ".fai")
    fai.write_text("\n".join(fai.read_text().splitlines()[:2]) + "\n")

    r = _run_cli(str(fasta), "esm2_8m", "--random_init", "--expect_sequences", "5",
                 "--output_hdf5_file", str(tmp_path / "o.h5"))
    assert r.returncode == 8, r.stderr
    assert "cohort size mismatch" in r.stderr
    assert not (tmp_path / "o.h5").exists(), "a short cohort must not leave an artifact"


def test_empty_cohort_is_not_a_success(tmp_path):
    """A zero-length .fai yields zero records; that used to exit 0."""
    fasta = tmp_path / "c.fasta"
    _write_fasta(fasta, 5)
    pytest.importorskip("pyfaidx").Fasta(str(fasta))
    Path(str(fasta) + ".fai").write_text("")

    r = _run_cli(str(fasta), "esm2_8m", "--random_init", "--expect_sequences", "5",
                 "--output_hdf5_file", str(tmp_path / "o.h5"))
    assert r.returncode != 0, "an empty cohort must never report success"
    assert not (tmp_path / "o.h5").exists()


def test_a_preexisting_index_is_left_for_the_other_tasks(tmp_path):
    """Deleting a shared index is what makes the race recur for a whole array."""
    fasta = tmp_path / "c.fasta"
    _write_fasta(fasta, 5)
    pytest.importorskip("pyfaidx").Fasta(str(fasta))
    fai = Path(str(fasta) + ".fai")
    assert fai.is_file()

    _run_cli(str(fasta), "esm2_8m", "--random_init", "--expect_sequences", "5",
             "--output_hdf5_file", str(tmp_path / "o.h5"))
    assert fai.is_file(), "a pre-existing shared index must survive the run"
