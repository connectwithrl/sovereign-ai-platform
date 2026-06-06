import pytest

from sovereign.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        serving_backend="echo",
        embedding_backend="hashing",
        embedding_dim=256,
        database_url=None,
        top_k=3,
        rerank=True,
        rerank_candidates=10,
    )


SAMPLE_DOCS = {
    "leave-policy": (
        "Annual leave policy. Full-time employees accrue thirty calendar days of paid "
        "annual leave each year. Leave requests must be submitted at least seven days in "
        "advance through the HR portal. Unused leave may be carried over up to a maximum "
        "of ten days into the following year."
    ),
    "data-classification": (
        "Data classification standard. Government data is classified as Public, Internal, "
        "Confidential, or Secret. Confidential and Secret data must remain within the "
        "sovereign cloud boundary and may never be sent to external model providers. All "
        "access to Secret data requires multi-factor authentication and is logged."
    ),
    "procurement": (
        "Procurement guidelines. Any purchase above fifty thousand dirhams requires three "
        "competitive quotations and approval from the department director. Contracts are "
        "reviewed annually by the procurement committee."
    ),
}