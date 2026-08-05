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

from transformers import BatchEncoding  # noqa: E402

from data_preparation.embeddings.embedding_generation import (  # noqa: E402
    generate_embeddings_batched,
)
from tests.conftest import tiny_esm  # noqa: E402

CLS, EOS, PAD = 0, 2, 1


class StubTokenizer:
    """Minimal stand-in: maps residues to ids, adds special tokens, right-pads.

    ``n_lead``/``n_trail`` mirror the family being simulated, so the same driver
    test can cover ProtT5/Ankh's trailing-only ``</s>`` and ProstT5's
    ``<AA2fold>`` prefix rather than only ESM's symmetric CLS/EOS.

    Returns a real ``BatchEncoding`` rather than a dict — the single-sequence
    path calls ``.to(device)`` on the result, which a plain dict does not have.
    """

    padding_side = "right"

    def __init__(self, n_lead: int = 1, n_trail: int = 1):
        self.n_lead = n_lead
        self.n_trail = n_trail

    def __call__(self, seqs, return_tensors=None, padding=True, **kwargs):
        if isinstance(seqs, str):
            seqs = [seqs]
        # `is_split_into_words=True` hands us lists of characters.
        flat = ["".join(s) if isinstance(s, list) else s for s in seqs]
        rows = [
            [CLS] * self.n_lead
            + [(ord(c) % 20) + 4 for c in s.replace(" ", "").replace("<AA2fold>", "")]
            + [EOS] * self.n_trail
            for s in flat
        ]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), PAD, dtype=torch.long)
        att = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            att[i, : len(r)] = 1
        return BatchEncoding({"input_ids": ids, "attention_mask": att})


@pytest.fixture(scope="module")
def tiny_model():
    return tiny_esm()


SEQS = [
    "MKTAYIAKQR",
    "MKV",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "MKTAYIAK",
    "MA",
]


def _unbatched(model, tokenizer, seqs, n_lead=1, n_trail=1):
    """The reference: one sequence at a time, slice off the special tokens, mean.

    Sorting by length is what bucketing does internally, so comparing every
    sequence against its own single-sequence reference is also the order check:
    a permutation shows up as a mismatch on the sequences sorting would move.
    """
    out = []
    for s in seqs:
        enc = tokenizer([s])
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state.squeeze(0)
        out.append(hidden[n_lead : hidden.shape[0] - n_trail, :].mean(dim=0).numpy())
    return out


@pytest.mark.parametrize(
    "family_key,n_lead,n_trail",
    # Every entry of SPECIAL_TOKEN_TRIM. Testing only esm_transformer left the
    # prot_t5/ankh (0, 1) trim and the prost_t5 <AA2fold> prefix unexercised —
    # SPECIAL_TOKEN_TRIM["prot_t5"] could have been (1, 1) with a green suite.
    [
        ("esm_transformer", 1, 1),
        ("prost_t5", 1, 1),
        ("prot_t5", 0, 1),
        ("ankh", 0, 1),
    ],
)
def test_batched_matches_unbatched_per_protein(tiny_model, family_key, n_lead, n_trail):
    from data_preparation.embeddings.embedding_generation import SPECIAL_TOKEN_TRIM

    assert SPECIAL_TOKEN_TRIM[family_key] == (n_lead, n_trail)

    tokenizer = StubTokenizer(n_lead=n_lead, n_trail=n_trail)
    expected = _unbatched(tiny_model, tokenizer, SEQS, n_lead, n_trail)

    got = generate_embeddings_batched(
        tiny_model,
        tokenizer,
        SEQS,
        family_key=family_key,
        device=torch.device("cpu"),
        token_budget=64,
    )

    assert len(got) == len(expected)
    for i, (g, e) in enumerate(zip(got, expected, strict=True)):
        assert g.shape == e.shape, f"sequence {i} shape {g.shape} != {e.shape}"
        assert np.allclose(g, e, atol=1e-5), (
            f"sequence {i} (len {len(SEQS[i])}) differs, max |d| = {np.abs(g - e).max():.2e}"
        )


def test_left_padding_is_rejected_rather_than_silently_pooling_the_wrong_tokens(
    tiny_model,
):
    """content_mask keeps [n_lead, length - n_trail), which assumes right padding.

    All four production tokenizers right-pad today, so this cannot fire — but a
    left-padding one would pool pad positions into the protein and still return
    plausible vectors, which is the one failure mode nothing downstream catches.
    """
    tokenizer = StubTokenizer()
    tokenizer.padding_side = "left"
    with pytest.raises(NotImplementedError, match="right padding"):
        generate_embeddings_batched(
            tiny_model, tokenizer, SEQS, family_key="esm_transformer",
            device=torch.device("cpu"), token_budget=64,
        )


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
