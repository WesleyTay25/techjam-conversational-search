"""In-memory dense semantic route. OWNER: Branch B.

TF-IDF -> TruncatedSVD -> L2-normalized float32 matrix, cosine by a single
numpy matmul. No external vector database, no network, ~50 MB resident for the
50k catalog. This is the arm that carries "for a wedding" -> formal/midi/silk
when no shared keyword exists.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.contracts import Candidate, DialogState, Product

# Situational phrase -> product-vocabulary expansion for the browsing track.
# Ablate before defending: if a line does not move the browsing hit rate on
# the dev slice, delete it rather than keep it for symmetry.
SCENARIO_LEXICON: dict[str, str] = {
    "wedding": "formal gown midi maxi chiffon satin elegant",
    "gym": "athletic moisture wicking performance spandex",
    "workout": "athletic moisture wicking performance spandex",
    "running": "moisture wicking breathable lightweight athletic",
    "hiking": "waterproof outdoor trail rugged durable",
    "office": "tailored professional blouse button front",
    "work": "tailored professional blouse button front",
    "beach": "linen breathable lightweight sandal",
    "date": "elegant flattering chic",
    "party": "sequin cocktail evening glam",
    "winter": "wool fleece thermal insulated warm",
    "summer": "lightweight breathable cotton sleeveless",
    "rain": "waterproof water resistant",
    "travel": "wrinkle resistant packable comfortable",
    "casual": "relaxed everyday comfortable cotton",
    "formal": "tailored elegant dressy",
    "outdoor": "durable waterproof rugged",
    "sleep": "soft cozy loungewear pajama",
    "interview": "tailored professional polished",
}


def _expand_scenarios(text: str) -> str:
    """Append product vocabulary for any situational phrase found in `text`."""
    lowered = text.lower()
    extra = [words for phrase, words in SCENARIO_LEXICON.items() if phrase in lowered]
    return text if not extra else text + " " + " ".join(extra)


def _build_query_text(state: DialogState) -> str:
    """Query text from dialog state, not the raw message.

    Weighted by `Slot.weight` so decayed slots contribute less and erased
    slots (weight 0.0) drop out entirely, plus a light pull from the
    anonymized profile summary.
    """
    parts: list[str] = []
    if state.category_tail:
        parts.append(state.category_tail)
    for slot in state.active_slots():
        repeats = max(1, round(slot.weight * 3))
        parts.extend([slot.value] * repeats)
    summary = state.user_profile.get("summary") if isinstance(state.user_profile, dict) else None
    if summary:
        parts.append(str(summary))
    return _expand_scenarios(" ".join(parts))


class DenseRoute:
    name = "dense"

    def __init__(self, products: dict[str, Product], dim: int = 256) -> None:
        self.asins: list[str] = list(products.keys())
        n_samples = len(self.asins)
        corpus = [products[asin].text_blob for asin in self.asins]

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=min(2, n_samples) if n_samples > 1 else 1,
            sublinear_tf=True,
        )
        tfidf = self._vectorizer.fit_transform(corpus)
        n_features = tfidf.shape[1]

        # TruncatedSVD requires n_components < min(n_samples, n_features);
        # the 50k-row catalog never hits this clamp, only the tiny test fixture does.
        self.dim = max(1, min(dim, n_features - 1, n_samples - 1))
        self._svd = TruncatedSVD(n_components=self.dim, random_state=0)
        reduced = self._svd.fit_transform(tfidf)

        self.matrix: np.ndarray = normalize(reduced).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        expanded = _expand_scenarios(text)
        tfidf = self._vectorizer.transform([expanded])
        reduced = self._svd.transform(tfidf)
        normed = normalize(reduced).astype(np.float32)
        return normed[0]

    def search(self, state: DialogState, limit: int) -> list[Candidate]:
        if limit <= 0:
            return []
        query_text = _build_query_text(state)
        query_vec = self.encode_query(query_text)
        scores = self.matrix @ query_vec

        depth = min(limit, len(self.asins))
        if depth >= len(scores):
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, depth - 1)[:depth]
            order = top[np.argsort(-scores[top])]

        return [
            Candidate(
                parent_asin=self.asins[i],
                score=float(scores[i]),
                route=self.name,
                evidence={"dense_score": float(scores[i])},
            )
            for i in order
        ]
