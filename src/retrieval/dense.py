"""In-memory dense semantic route. OWNER: Branch B.

TF-IDF -> TruncatedSVD -> L2-normalized float32 matrix, cosine by a single
numpy matmul. No external vector database, no network, ~50 MB resident for the
50k catalog. This is the arm that carries "for a wedding" -> formal/midi/silk
when no shared keyword exists.
"""

from __future__ import annotations

from src.contracts import Candidate, DialogState, Product


class DenseRoute:
    name = "dense"

    def __init__(self, products: dict[str, Product], dim: int = 256) -> None:
        raise NotImplementedError("Branch B")

    def search(self, state: DialogState, limit: int) -> list[Candidate]:
        raise NotImplementedError("Branch B")

    def encode_query(self, text: str):
        raise NotImplementedError("Branch B")
