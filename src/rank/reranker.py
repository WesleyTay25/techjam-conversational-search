"""Deterministic offline feature reranker. OWNER: Branch D.

This is the layer that carries the official run when the organizer disables the
network, so it touches no remote service and uses no RNG. It takes the fused
candidate set, scores every row on a small linear model over interpretable
features, then applies two ordering rules on top: a mild penalty for products
already shown without converting, and light MMR diversification on the browsing
track only.

Feature design follows the brief's table. `constraint_exact_hits` is expected to
dominate and is deliberately left un-normalised so it can; the route scores and
the rating prior are only meaningful relative to the candidate set, so they are
standardised across it before the linear combination.

Weights ship hand-tuned (`DEFAULT_WEIGHTS`). `fit_weights` fits them with
logistic regression and 5-fold CV against a feature matrix built offline by
`tools/`; the brief caps the fitted magnitudes and asks for the CV spread in the
PR, because 200 public sessions is a small sample and the private set is four
times larger and generated differently.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Iterable, Sequence

from src.contracts import (
    COLOR_RE,
    MATERIAL_RE,
    Candidate,
    DialogState,
    Product,
    Slot,
    TRACK_BROWSING,
    normalize_key,
)

__all__ = ["FeatureReranker", "DEFAULT_WEIGHTS", "fit_weights"]

# Hand-tuned starting point. Signs matter more than magnitudes here: exact
# constraint hits and slot coverage pull up, price distance / erased-slot
# agreement / re-showing pull down. Fit against real sessions before trusting
# the magnitudes (see `fit_weights`).
DEFAULT_WEIGHTS: dict[str, float] = {
    "constraint_exact_hits": 3.2,
    "constraint_coverage": 2.1,
    "constraint_jaccard_mean": 1.3,
    "tail_exact": 1.6,
    "price_delta": -1.1,
    "color_match": 0.9,
    "material_match": 0.9,
    "bm25_z": 0.7,
    "dense_cos": 0.7,
    "fusion_z": 1.0,
    "profile_tag_overlap": 0.5,
    "rating_prior": 0.15,
    "erased_slot_match": -2.6,
    "already_shown": -0.35,
}

# Columns that only mean something relative to the current candidate set.
_STANDARDISE = ("bm25_z", "dense_cos", "fusion_z", "rating_prior")

_WORD = re.compile(r"[a-z0-9]+")
_PRICE = re.compile(r"\d+(?:\.\d+)?")
_STOP = {"the", "a", "an", "for", "with", "and", "of", "in", "to", "is", "or"}

_MMR_LAMBDA = 0.72
_MMR_WINDOW = 20
_MMR_DEPTH = 10


def _standardise(rows: list[tuple[Candidate, dict[str, float]]], key: str) -> None:
    values = [feats.get(key, 0.0) for _, feats in rows]
    if not values:
        return
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    for _, feats in rows:
        feats[key] = 0.0 if sd == 0.0 else (feats.get(key, 0.0) - mean) / sd


class FeatureReranker:
    def __init__(self, products: dict[str, Product], weights: dict[str, float] | None = None) -> None:
        self.products = products
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    # -- public API ---------------------------------------------------------

    def features(self, state: DialogState, candidate: Candidate) -> dict[str, float]:
        """Per-candidate feature vector. Route scores and the rating prior are
        returned raw here; `rerank` standardises them across the set."""
        product = self.products.get(candidate.parent_asin)
        if product is None:
            return {}

        disclosed_keys = [normalize_key(text) for text in state.raw_constraints]
        blob_tokens = set(_WORD.findall(product.text_blob))

        feats: dict[str, float] = {}
        feats["constraint_exact_hits"] = float(
            sum(1 for key in disclosed_keys if key in product.constraint_keys)
        )
        feats["constraint_coverage"] = self._hard_slot_coverage(state, product)
        feats["constraint_jaccard_mean"] = self._constraint_containment(state, blob_tokens)
        feats["tail_exact"] = self._tail_score(state, product)
        feats["price_delta"] = self._price_delta(state, product)

        wanted_color = self._disclosed_value("color", state)
        wanted_material = self._disclosed_value("material", state)
        feats["color_match"] = 1.0 if wanted_color and product.color == wanted_color else 0.0
        feats["material_match"] = 1.0 if wanted_material and product.material == wanted_material else 0.0

        feats["bm25_z"] = _route_score(candidate, ("bm25", "lexical"))
        feats["dense_cos"] = _route_score(candidate, ("dense", "semantic"))
        feats["fusion_z"] = float(candidate.score)

        feats["profile_tag_overlap"] = self._profile_overlap(state, product)
        feats["rating_prior"] = math.log1p(product.rating_number) * (product.average_rating or 0.0)
        feats["erased_slot_match"] = float(
            sum(1 for slot in _erased_slots(state) if self._slot_satisfied(slot, product))
        )
        feats["already_shown"] = 1.0 if candidate.parent_asin in state.shown else 0.0
        return feats

    def rerank(self, state: DialogState, candidates: Sequence[Candidate]) -> list[Candidate]:
        # Work on copies so the call is pure: the orchestrator rebuilds the
        # candidate list every turn and the determinism test reuses inputs.
        copies = [replace(cand, evidence=dict(cand.evidence)) for cand in candidates]
        if not copies:
            return []
        rows = [(cand, self.features(state, cand)) for cand in copies]

        for column in _STANDARDISE:
            _standardise(rows, column)

        for cand, feats in rows:
            cand.score = sum(self.weights.get(name, 0.0) * value for name, value in feats.items())
            cand.evidence["rerank_score"] = cand.score

        # Stable primary sort; parent_asin breaks ties so two runs agree exactly.
        ordered = sorted((cand for cand, _ in rows), key=lambda c: (-c.score, c.parent_asin))

        if state.track == TRACK_BROWSING:
            ordered = self._diversify(ordered)
        return ordered

    # -- feature helpers ----------------------------------------------

    def _hard_slot_coverage(self, state: DialogState, product: Product) -> float:
        hard = [slot for slot in state.active_slots() if slot.hard]
        if not hard:
            return 0.0
        return sum(1 for slot in hard if self._slot_satisfied(slot, product)) / len(hard)

    def _constraint_containment(self, state: DialogState, blob_tokens: set[str]) -> float:
        """Fraction of each disclosed constraint's tokens found in the product
        text, averaged. The paraphrase-robust counterpart to the exact-key
        count -- it still fires when the private simulator rewords a constraint."""
        ratios: list[float] = []
        for raw in state.raw_constraints:
            tokens = {t for t in _WORD.findall(raw.lower()) if t not in _STOP}
            if tokens:
                ratios.append(len(tokens & blob_tokens) / len(tokens))
        return sum(ratios) / len(ratios) if ratios else 0.0

    def _tail_score(self, state: DialogState, product: Product) -> float:
        if not state.category_tail:
            return 0.0
        if product.category_tail == state.category_tail:
            return 1.0
        want = set(state.category_tail.lower().split())
        have = set(product.category_tail.lower().split())
        return len(want & have) / len(want | have) if want and have else 0.0

    def _price_delta(self, state: DialogState, product: Product) -> float:
        if not state.price_target or product.price is None:
            return 0.0
        return min(2.0, abs(product.price - state.price_target) / state.price_target)

    def _profile_overlap(self, state: DialogState, product: Product) -> float:
        tags = state.user_profile.get("preference_tags") or []
        if not tags:
            return 0.0
        return sum(1 for tag in tags if str(tag).lower() in product.text_blob) / len(tags)

    def _disclosed_value(self, attribute: str, state: DialogState) -> str | None:
        pattern = COLOR_RE if attribute == "color" else MATERIAL_RE
        for slot in state.active_slots():
            if slot.attribute == attribute:
                found = pattern.search(slot.value)
                return found.group(1).lower() if found else slot.value.strip().lower() or None
        for raw in state.raw_constraints:
            found = pattern.search(raw)
            if found:
                return found.group(1).lower()
        return None

    def _slot_satisfied(self, slot: Slot, product: Product) -> bool:
        value = slot.value.lower().strip()
        if not value:
            return False
        if normalize_key(slot.value) in product.constraint_keys:
            return True
        if slot.attribute == "color":
            return product.color is not None and product.color in value
        if slot.attribute == "material":
            return product.material is not None and product.material in value
        if slot.attribute == "budget" and product.price is not None:
            match = _PRICE.search(value)
            if match:
                target = float(match.group())
                return target > 0 and abs(product.price - target) <= 0.15 * target
        tokens = [t for t in _WORD.findall(value) if t not in _STOP]
        return bool(tokens) and all(token in product.text_blob for token in tokens)

    # -- diversification ---------------------------------------------

    def _diversify(self, ordered: list[Candidate]) -> list[Candidate]:
        """Light MMR over the head of the browsing-track list. Ten colourways of
        one dress is one guess, not ten; on the buying track we skip this
        because precision beats coverage there."""
        if len(ordered) <= 2:
            return ordered
        window = ordered[:_MMR_WINDOW]
        tail = ordered[_MMR_WINDOW:]
        scores = [c.score for c in window]
        low, high = min(scores), max(scores)
        norm = [1.0 if high == low else (s - low) / (high - low) for s in scores]

        picked = [0]
        picked_set = {0}
        while len(picked) < min(_MMR_DEPTH, len(window)):
            best_index = None
            best_value = None
            for i, cand in enumerate(window):
                if i in picked_set:
                    continue
                similarity = max(self._similarity(cand, window[j]) for j in picked)
                value = _MMR_LAMBDA * norm[i] - (1.0 - _MMR_LAMBDA) * similarity
                if best_value is None or value > best_value:
                    best_index, best_value = i, value
            picked.append(best_index)
            picked_set.add(best_index)

        reordered = [window[i] for i in picked]
        leftovers = [cand for i, cand in enumerate(window) if i not in picked_set]
        return reordered + leftovers + tail

    def _similarity(self, left: Candidate, right: Candidate) -> float:
        a = self.products.get(left.parent_asin)
        b = self.products.get(right.parent_asin)
        if a is None or b is None:
            return 0.0
        score = 0.0
        if a.category_tail and a.category_tail == b.category_tail:
            score += 0.5
        if a.color and a.color == b.color:
            score += 0.25
        if a.material and a.material == b.material:
            score += 0.25
        return score


def _route_score(candidate: Candidate, names: tuple[str, ...]) -> float:
    for name in names:
        if name in candidate.evidence:
            return float(candidate.evidence[name])
    return 0.0


def _erased_slots(state: DialogState) -> Iterable[Slot]:
    for group in state.slots.values():
        for slot in group:
            if slot.weight == 0.0:
                yield slot


def fit_weights(
    feature_rows: Sequence[dict[str, float]],
    labels: Sequence[int],
    *,
    l2: float = 1.0,
    cap: float = 10.0,
    folds: int = 5,
    seed: int = 0,
) -> tuple[dict[str, float], float, float]:
    """Fit the linear weights with logistic regression and k-fold CV.

    `feature_rows` / `labels` come from a feature matrix built offline by
    `tools/` over the public sessions (row per candidate, label = is-target).
    Returns (weights, cv_mean, cv_sd) using ROC-AUC; report the spread in the PR
    and ship `DEFAULT_WEIGHTS` instead if the variance is high.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    names = sorted({name for row in feature_rows for name in row})
    matrix = [[row.get(name, 0.0) for name in names] for row in feature_rows]
    targets = list(labels)

    model = LogisticRegression(C=1.0 / l2, max_iter=1000)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, matrix, targets, cv=splitter, scoring="roc_auc")

    model.fit(matrix, targets)
    weights = {
        name: max(-cap, min(cap, float(coef)))
        for name, coef in zip(names, model.coef_[0])
    }
    return weights, float(scores.mean()), float(scores.std())
