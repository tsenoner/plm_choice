"""The random-init baseline must be an untrained architecture, not a broken one.

`--random_init` answers reviewer R1.9 ("random noise is not the right baseline —
use a frozen, randomly-initialized version of the same models"), so the whole
point is that the embeddings still carry the architectural prior: attention,
layer norms and positional encoding shaping residues in context, with no learned
biology.

The original implementation overwrote *every* parameter with ``N(0, 0.02)``,
which sets LayerNorm gains to ~0 and biases to ~0.02 instead of 1.0 and 0. That
collapses the hidden-state magnitude by ~34x and drives the mean residue-residue
correlation from +0.69 to +0.96 — every residue ends up with nearly the same
vector, so the mean-pooled protein embedding degenerates to a constant plus
noise. It would have looked like a working baseline and reported a meaningless
null.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoModel  # noqa: E402
from transformers.models.esm.configuration_esm import EsmConfig  # noqa: E402

from data_preparation.embeddings.embedding_generation import _reinit_weights  # noqa: E402


def _tiny_config() -> EsmConfig:
    """A 4-layer ESM, small enough to build in a fraction of a second."""
    return EsmConfig(
        vocab_size=33,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        position_embedding_type="rotary",
        pad_token_id=1,
    )


@pytest.fixture
def reinitialised_model():
    torch.manual_seed(0)
    model = AutoModel.from_config(_tiny_config()).eval()
    _reinit_weights(model, seed=42)
    return model


def _hidden_states(model):
    ids = torch.randint(4, 30, (1, 60))
    with torch.no_grad():
        return model(ids).last_hidden_state


def test_layernorm_gains_stay_at_one(reinitialised_model):
    gains = [
        p for n, p in reinitialised_model.named_parameters() if "LayerNorm.weight" in n
    ]
    assert gains, "no LayerNorm weights found — the test is not exercising anything"
    for gain in gains:
        assert torch.allclose(gain, torch.ones_like(gain)), (
            "LayerNorm gains were overwritten with noise; this is the defect that "
            "collapses the embedding space"
        )


def test_biases_stay_at_zero(reinitialised_model):
    biases = [p for n, p in reinitialised_model.named_parameters() if n.endswith(".bias")]
    assert biases
    for bias in biases:
        assert torch.allclose(bias, torch.zeros_like(bias))


def test_linear_weights_are_actually_randomised(reinitialised_model):
    """The counterpart: weights must NOT be left at their prior values."""
    weights = [
        p
        for name, p in reinitialised_model.named_parameters()
        if name.endswith(".weight") and "LayerNorm" not in name and p.dim() == 2
    ]
    assert weights
    stds = torch.stack([w.std() for w in weights])
    # initializer_range is 0.02; allow a generous band but reject "all zeros"
    # and "still pretrained-scale".
    assert 0.005 < stds.mean() < 0.06, f"unexpected weight scale: {stds.mean():.4f}"


def test_embeddings_remain_contextual(reinitialised_model):
    """The load-bearing property: residues must not all collapse onto one vector."""
    hidden = _hidden_states(reinitialised_model)
    correlations = np.corrcoef(hidden[0].numpy())
    off_diagonal = correlations[np.triu_indices_from(correlations, 1)]
    mean_corr = off_diagonal.mean()
    assert mean_corr < 0.90, (
        f"mean residue-residue correlation {mean_corr:+.3f} is too high — the "
        f"per-residue vectors have collapsed, so mean-pooling yields a near-constant "
        f"protein embedding (the broken scheme measured +0.96)"
    )


def test_hidden_states_have_a_healthy_magnitude(reinitialised_model):
    hidden = _hidden_states(reinitialised_model)
    magnitude = float(hidden.abs().mean())
    assert magnitude > 0.1, (
        f"hidden-state magnitude {magnitude:.2e} is vanishing; the broken scheme "
        f"measured 2.4e-02 against ~8.0e-01 for a healthy init"
    )


def test_reinit_is_reproducible_for_a_given_seed():
    """--random_seed exists so D-6's 3-seed mean+-sd is a flag, not a code change."""

    def build(seed):
        torch.manual_seed(0)
        model = AutoModel.from_config(_tiny_config()).eval()
        _reinit_weights(model, seed=seed)
        return next(
            p for n, p in model.named_parameters() if n.endswith(".weight") and p.dim() == 2
        ).clone()

    assert torch.equal(build(42), build(42)), "same seed must give the same weights"
    assert not torch.equal(build(0), build(1)), "different seeds must differ"
