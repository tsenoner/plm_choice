"""The batched driver must reproduce the per-sequence path it replaces.

This is the test that actually protects the numbers. The primitives are unit
tested elsewhere; here a real transformer is run both ways over the same
sequences and the resulting protein embeddings are compared.

No network: the model is built from a tiny config and the tokenizer is a stub
implementing the slice of the HuggingFace interface the driver uses. The
tokenizer is not what is under test — the batching, masking and ordering are.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoModel, BatchEncoding  # noqa: E402
from transformers.models.esm.configuration_esm import EsmConfig  # noqa: E402

from data_preparation.embeddings.embedding_generation import (  # noqa: E402
    generate_embeddings_batched,
)

CLS, EOS, PAD = 0, 2, 1


class StubTokenizer:
    """Minimal stand-in: maps residues to ids, wraps in CLS/EOS, right-pads.

    Returns a real ``BatchEncoding`` rather than a dict — the single-sequence
    path calls ``.to(device)`` on the result, which a plain dict does not have.
    """

    def __call__(self, seqs, return_tensors=None, padding=True, **kwargs):
        if isinstance(seqs, str):
            seqs = [seqs]
        # `is_split_into_words=True` hands us lists of characters.
        flat = ["".join(s) if isinstance(s, list) else s for s in seqs]
        rows = [[CLS] + [(ord(c) % 20) + 4 for c in s.replace(" ", "")] + [EOS] for s in flat]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), PAD, dtype=torch.long)
        att = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            att[i, : len(r)] = 1
        return BatchEncoding({"input_ids": ids, "attention_mask": att})


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = EsmConfig(
        vocab_size=33,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        position_embedding_type="rotary",
        pad_token_id=PAD,
    )
    return AutoModel.from_config(cfg).eval()


SEQS = [
    "MKTAYIAKQR",
    "MKV",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "MKTAYIAK",
    "MA",
]


def _unbatched(model, tokenizer, seqs):
    """The reference: one sequence at a time, slice off CLS/EOS, then mean."""
    out = []
    for s in seqs:
        enc = tokenizer([s])
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state.squeeze(0)
        out.append(hidden[1:-1, :].mean(dim=0).numpy())
    return out


def test_batched_matches_unbatched_per_protein(tiny_model):
    tokenizer = StubTokenizer()
    expected = _unbatched(tiny_model, tokenizer, SEQS)

    got = generate_embeddings_batched(
        tiny_model,
        tokenizer,
        SEQS,
        family_key="esm_transformer",
        device=torch.device("cpu"),
        token_budget=64,
    )

    assert len(got) == len(expected)
    for i, (g, e) in enumerate(zip(got, expected, strict=True)):
        assert g.shape == e.shape, f"sequence {i} shape {g.shape} != {e.shape}"
        assert np.allclose(g, e, atol=1e-5), (
            f"sequence {i} (len {len(SEQS[i])}) differs, max |d| = {np.abs(g - e).max():.2e}"
        )


def test_batched_preserves_input_order(tiny_model):
    """Bucketing sorts by length internally; the caller must not see that."""
    tokenizer = StubTokenizer()
    got = generate_embeddings_batched(
        tiny_model, tokenizer, SEQS, family_key="esm_transformer",
        device=torch.device("cpu"), token_budget=64,
    )
    # The two identical-prefix sequences of different length must not be swapped:
    # compare against single-sequence extraction for the shortest input, which
    # sorting would otherwise have moved to the front.
    enc = tokenizer([SEQS[-1]])
    with torch.no_grad():
        hidden = tiny_model(**enc).last_hidden_state.squeeze(0)
    expected_last = hidden[1:-1, :].mean(dim=0).numpy()
    assert np.allclose(got[-1], expected_last, atol=1e-5)


def test_batch_size_does_not_change_the_result(tiny_model):
    """Throughput knob must be a pure performance dial, not a numerical one."""
    tokenizer = StubTokenizer()
    small = generate_embeddings_batched(
        tiny_model, tokenizer, SEQS, family_key="esm_transformer",
        device=torch.device("cpu"), token_budget=16,
    )
    large = generate_embeddings_batched(
        tiny_model, tokenizer, SEQS, family_key="esm_transformer",
        device=torch.device("cpu"), token_budget=4096,
    )
    for i, (a, b) in enumerate(zip(small, large, strict=True)):
        assert np.allclose(a, b, atol=1e-5), f"sequence {i} changed with token_budget"


def test_native_esm_families_are_rejected_rather_than_silently_wrong(tiny_model):
    """ESM-3/ESM-C use their own encode() API; batching them is not implemented."""
    with pytest.raises(NotImplementedError):
        generate_embeddings_batched(
            tiny_model, StubTokenizer(), SEQS, family_key="native_esmc",
            device=torch.device("cpu"), token_budget=64,
        )
