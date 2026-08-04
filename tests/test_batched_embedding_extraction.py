"""Batched extraction must produce what the batch-size-1 path produces.

Embedding 481,320 proteins x 12 models x 3 seeds one sequence at a time, in fp32,
with no autocast is 300-900 GPU-h; batched with length bucketing it is 7-13. That
is the single biggest lever on Track B4's cost.

It is also the easiest place to corrupt the numbers invisibly. A padded batch has
two hazards the unbatched path simply does not have:

* pooling over padding — a plain ``.mean(dim=1)`` averages the pad rows in, and
  the error grows the shorter a protein is relative to its batch's longest member;
* special tokens — every family strips a different number of leading/trailing
  tokens, and with right-padding the trailing one is no longer at index -1.

Both produce plausible-looking vectors. These tests pin batched output against the
unbatched path it replaces, which is the only check that would actually catch it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from data_preparation.embeddings.embedding_generation import (  # noqa: E402
    content_mask,
    masked_mean_pool,
)

# --------------------------------------------------------------------------- #
#                  special-token stripping under right-padding
# --------------------------------------------------------------------------- #


def test_content_mask_drops_leading_and_trailing_special_tokens():
    """ESM: <cls> ... <eos>, i.e. one leading and one trailing token."""
    # Two sequences: 3 real residues and 1 real residue, each wrapped in cls/eos.
    attention = torch.tensor(
        [
            [1, 1, 1, 1, 1],  # cls r r r eos
            [1, 1, 1, 0, 0],  # cls r eos pad pad
        ]
    )
    mask = content_mask(attention, n_lead=1, n_trail=1)

    assert mask.tolist() == [
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
    ]


def test_content_mask_finds_the_trailing_token_per_sequence_not_at_index_minus_one():
    """The regression this exists to prevent.

    With right-padding the </s> of a short sequence sits in the middle of the row.
    Slicing [..., :-1] would strip a pad and keep the real special token.
    """
    attention = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    mask = content_mask(attention, n_lead=0, n_trail=1)

    assert mask.tolist() == [[1, 1, 1, 0], [1, 0, 0, 0]]


def test_content_mask_handles_prot_t5_style_trailing_only():
    attention = torch.tensor([[1, 1, 1, 1, 1]])
    assert content_mask(attention, n_lead=0, n_trail=1).tolist() == [[1, 1, 1, 1, 0]]


def test_content_mask_rejects_a_sequence_with_nothing_left():
    """A sequence shorter than its own special tokens must fail, not pool nothing."""
    attention = torch.tensor([[1, 1]])  # cls + eos, zero residues
    with pytest.raises(ValueError):
        content_mask(attention, n_lead=1, n_trail=1)


# --------------------------------------------------------------------------- #
#            batched pooling == the unbatched slice-then-mean it replaces
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n_lead,n_trail",
    [(1, 1), (0, 1)],  # esm/prost_t5 style, and prot_t5/ankh style
    ids=["cls_and_eos", "eos_only"],
)
def test_batched_pooling_matches_unbatched_slicing(n_lead, n_trail):
    """The exact equivalence the batched path has to preserve."""
    torch.manual_seed(0)
    lengths = [6, 3, 9, 4]  # real token counts INCLUDING special tokens
    dim = 5
    width = max(lengths)

    hidden = torch.zeros(len(lengths), width, dim)
    attention = torch.zeros(len(lengths), width, dtype=torch.long)
    expected = []
    for i, n in enumerate(lengths):
        row = torch.randn(n, dim)
        hidden[i, :n] = row
        attention[i, :n] = 1
        # What the unbatched path computes: slice off the special tokens, then mean.
        sliced = row[n_lead : n - n_trail] if n_trail else row[n_lead:]
        expected.append(sliced.mean(dim=0))

    pooled = masked_mean_pool(hidden, content_mask(attention, n_lead, n_trail))

    for i, want in enumerate(expected):
        assert torch.allclose(pooled[i], want, atol=1e-6), f"sequence {i} differs"


def test_batching_does_not_depend_on_who_shares_the_batch():
    """A protein's embedding must not change with batch composition.

    If it does, results depend on sort order and are not reproducible.
    """
    torch.manual_seed(1)
    dim = 4
    short = torch.randn(3, dim)

    alone = masked_mean_pool(
        short.unsqueeze(0), content_mask(torch.ones(1, 3, dtype=torch.long), 1, 1)
    )

    padded = torch.zeros(1, 8, dim)
    padded[0, :3] = short
    att = torch.zeros(1, 8, dtype=torch.long)
    att[0, :3] = 1
    with_padding = masked_mean_pool(padded, content_mask(att, 1, 1))

    assert torch.allclose(alone[0], with_padding[0], atol=1e-6)


def test_pooling_is_stable_in_bfloat16_within_tolerance():
    """bf16 is the throughput lever; it must not move the vector meaningfully."""
    torch.manual_seed(2)
    hidden = torch.randn(2, 10, 16)
    att = torch.ones(2, 10, dtype=torch.long)
    mask = content_mask(att, 1, 1)

    fp32 = masked_mean_pool(hidden, mask)
    bf16 = masked_mean_pool(hidden.to(torch.bfloat16), mask).to(torch.float32)

    # bf16 carries ~3 decimal digits; assert the direction is preserved rather
    # than pretending the bits match.
    cos = torch.nn.functional.cosine_similarity(fp32, bf16, dim=-1)
    assert bool((cos > 0.999).all()), f"cosine dropped to {cos.tolist()}"


def test_length_buckets_respect_the_token_budget():
    from data_preparation.embeddings.embedding_generation import length_buckets

    seqs = ["A" * n for n in (10, 12, 11, 100, 5)]
    batches = length_buckets(seqs, token_budget=30)

    # Every batch must fit: len(batch) * longest-in-batch <= budget, except a
    # single sequence that exceeds the budget on its own.
    for batch in batches:
        longest = max(len(seqs[i]) for i in batch)
        assert len(batch) == 1 or len(batch) * longest <= 30, (
            f"batch {batch} exceeds budget"
        )

    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(seqs))), "every sequence exactly once"


def test_length_buckets_group_similar_lengths_together():
    """The point of bucketing: minimise padding waste."""
    from data_preparation.embeddings.embedding_generation import length_buckets

    seqs = ["A" * n for n in (10, 500, 11, 12)]
    batches = length_buckets(seqs, token_budget=10_000)
    bucket_of = {i: b for b, batch in enumerate(batches) for i in batch}

    # The three ~10-residue sequences belong together; the 500 should not drag
    # them into a 500-wide padded batch.
    assert bucket_of[0] == bucket_of[2] == bucket_of[3]
    assert bucket_of[1] != bucket_of[0]
