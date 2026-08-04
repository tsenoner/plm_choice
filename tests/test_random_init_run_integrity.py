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
    _reinit_weights,
    default_output_path,
    masked_mean_pool,
    process_sequences_and_save,
    should_apply_post_load_hook,
)

# --------------------------------------------------------------------------- #
#            the random arm must run at its twin's compute precision
# --------------------------------------------------------------------------- #


def test_post_load_hook_applies_to_random_init_too():
    """`pretrained - random_init` is only a statement about weights.

    ProtT5/ProstT5 carry a `.half()` post-load hook, so skipping it for the
    random arm means the untrained model computes in fp32 while its pretrained
    twin computed in fp16. That difference lands directly in the headline
    difference and is indistinguishable from a pretraining effect.
    """
    assert should_apply_post_load_hook(post_load_hook=lambda m: None, random_init=True)


def test_post_load_hook_still_applies_to_pretrained():
    assert should_apply_post_load_hook(post_load_hook=lambda m: None, random_init=False)


def test_no_hook_means_nothing_to_apply():
    assert not should_apply_post_load_hook(post_load_hook=None, random_init=True)
    assert not should_apply_post_load_hook(post_load_hook=None, random_init=False)


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


def test_pretrained_run_may_still_resume(tmp_path):
    """Resuming a long pretrained run is a feature and must keep working.

    Deliberately no read-back here. h5py/HDF5 1.12 on macOS will not reopen a
    file read-only in the *same* process that just held it open for write, even
    after a clean close (a fresh process reads it fine, so the file itself is
    sound). Asserting on the return value instead proves what we actually care
    about: append mode saw the stored id and skipped it, rather than truncating.
    """
    h5py = pytest.importorskip("h5py")
    existing = tmp_path / "sprot_esm2_650m.h5"
    with h5py.File(existing, "w") as fh:
        fh.create_dataset("P12345", data=np.zeros(4, dtype=np.float32))

    # Feeding back an id the file already holds exercises the resume path: with
    # "w-" this would raise, and with "w" the dataset would be gone.
    n_done = process_sequences_and_save(
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
#                    batched pooling must ignore padding
# --------------------------------------------------------------------------- #


def test_masked_mean_pool_ignores_padding():
    """A plain .mean() over a padded batch is wrong, and worse for short proteins."""
    # Two sequences of real length 2 and 4, padded to 4.
    hidden = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    pooled = masked_mean_pool(hidden, mask)

    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(pooled[1], torch.tensor([2.5, 2.5]))


def test_masked_mean_pool_matches_unbatched_mean():
    """The batched path must reproduce the batch-size-1 result it replaces."""
    torch.manual_seed(0)
    lengths = [3, 7, 5]
    dim = 6
    padded = torch.zeros(len(lengths), max(lengths), dim)
    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    singles = []
    for i, n in enumerate(lengths):
        seq = torch.randn(n, dim)
        padded[i, :n] = seq
        mask[i, :n] = 1
        singles.append(seq.mean(dim=0))

    pooled = masked_mean_pool(padded, mask)

    for i, expected in enumerate(singles):
        assert torch.allclose(pooled[i], expected, atol=1e-6), f"row {i} differs"


def test_masked_mean_pool_rejects_an_all_padding_row():
    """Silently returning 0/0 = nan would poison the HDF5 rather than fail."""
    hidden = torch.zeros(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        masked_mean_pool(hidden, mask)
