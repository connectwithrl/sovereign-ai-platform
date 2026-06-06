"""Seed the running API with the sample corpus.

Usage:  python scripts/seed.py            # targets http://localhost:8000
        API_URL=http://host:8000 python scripts/seed.py
"""

from __future__ import annotations

import os
import sys

import httpx

# Allow running from the repo root without installing.
sys.path.insert(0, os.path.dirname(__file__))
from sample_data import SAMPLE_DOCS  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


def main() -> None:
    with httpx.Client(timeout=30) as client:
        client.get(f"{API_URL}/healthz").raise_for_status()
        for doc_id, text in SAMPLE_DOCS.items():
            r = client.post(
                f"{API_URL}/v1/ingest",
                json={"text": text, "source": doc_id, "doc_id": doc_id},
            )
            r.raise_for_status()
            body = r.json()
            print(f"  ingested {doc_id:<20} -> {body['chunks']} chunk(s)")
        total = client.get(f"{API_URL}/healthz").json()["documents"]
    print(f"Done. Knowledge base now holds {total} chunk(s).")


if __name__ == "__main__":
    main()