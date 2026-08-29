"""Catalog loading and normalization. LEAD-OWNED, landed on D0 so that no
feature branch is blocked waiting for `Product`.

Read-only: we parse the frozen JSONL once and derive indices. We never write
back, never synthesise ASINs, and never mutate a row.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.contracts import (
    COLOR_RE,
    MATERIAL_RE,
    Product,
    clean_constraint,
    coarse_category,
    flatten_values,
    normalize_key,
)

_TEXT_FIELDS = ("title", "features", "details", "description", "categories", "store")


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _as_price(value: object) -> float | None:
    try:
        price = float(str(value).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def build_product(row: dict) -> Product:
    """Normalize one catalog row into the frozen `Product` record.

    `constraint_keys` is the important derived field: it holds the flattened
    features and details strings under exactly the transform the simulator
    applies before it speaks them aloud, so a disclosed constraint becomes an
    O(1) hash lookup rather than a fuzzy text search.
    """
    features = tuple(str(v) for v in (row.get("features") or []) if v not in (None, ""))
    description = tuple(str(v) for v in (row.get("description") or []) if v not in (None, ""))
    categories = tuple(str(v) for v in (row.get("categories") or []) if v not in (None, ""))
    raw_details = row.get("details") or {}
    details = {str(k): str(v) for k, v in raw_details.items()} if isinstance(raw_details, dict) else {}

    blob = " ".join(_as_text(row.get(field)) for field in _TEXT_FIELDS).lower()

    keys = {
        normalize_key(clean_constraint(item))
        for item in (*flatten_values(list(features)), *flatten_values(details))
        if clean_constraint(item)
    }

    material = MATERIAL_RE.search(blob)
    color = COLOR_RE.search(blob)

    try:
        rating_number = int(row.get("rating_number") or 0)
    except (TypeError, ValueError):
        rating_number = 0
    try:
        average_rating = float(row["average_rating"]) if row.get("average_rating") is not None else None
    except (TypeError, ValueError):
        average_rating = None

    return Product(
        parent_asin=str(row["parent_asin"]),
        title=str(row.get("title") or ""),
        features=features,
        description=description,
        categories=categories,
        details=details,
        store=str(row.get("store") or ""),
        price=_as_price(row.get("price")),
        average_rating=average_rating,
        rating_number=rating_number,
        text_blob=blob,
        constraint_keys=frozenset(keys),
        category_tail=coarse_category(categories),
        material=material.group(1).lower() if material else None,
        color=color.group(1).lower() if color else None,
    )


def load_catalog(path: str | Path) -> dict[str, Product]:
    """Parse the frozen catalog into an ordered {parent_asin: Product} map."""
    products: dict[str, Product] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = build_product(json.loads(line))
            products[product.parent_asin] = product
    return products
