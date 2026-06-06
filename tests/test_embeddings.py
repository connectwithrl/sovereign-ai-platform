import numpy as np

from sovereign.embeddings import HashingEmbedder


def test_deterministic_and_normalised():
    emb = HashingEmbedder(dim=128)
    a = emb.embed(["data sovereignty in the cloud"])
    b = emb.embed(["data sovereignty in the cloud"])
    assert a.shape == (1, 128)
    np.testing.assert_array_equal(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-6


def test_similar_text_scores_higher_than_unrelated():
    emb = HashingEmbedder(dim=512)
    q = emb.embed(["annual leave accrual policy"])[0]
    related = emb.embed(["employees accrue annual leave each year"])[0]
    unrelated = emb.embed(["procurement requires competitive quotations"])[0]
    assert float(q @ related) > float(q @ unrelated)


def test_empty_batch():
    emb = HashingEmbedder(dim=64)
    assert emb.embed([]).shape == (0, 64)