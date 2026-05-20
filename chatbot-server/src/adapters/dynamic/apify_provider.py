from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from src.models.schemas import ProductRecord, QueryFilters

logger = logging.getLogger(__name__)


class ApifyProvider:  # pragma: no cover
    """
    Minimal Apify integration.

    This adapter assumes you have an Apify Actor that returns a dataset of product objects.
    We keep it generic so you can plug in Amazon/Flipkart actors.
    """

    def __init__(  # pragma: no cover
        self,
        token: Optional[str] = None,
        actor_id: Optional[str] = None,
        timeout_s: int = 60,
    ) -> None:
        self.token = token or os.getenv("APIFY_TOKEN", "")
        self.actor_id = actor_id or os.getenv("APIFY_ACTOR_ID", "")
        self.timeout_s = timeout_s

    def is_configured(self) -> bool:  # pragma: no cover
        return bool(self.token and self.actor_id)

    def fetch_products(self, query: str, filters: QueryFilters, limit: int = 30) -> List[ProductRecord]:  # pragma: no cover
        if not self.is_configured():
            raise RuntimeError("ApifyProvider not configured. Set APIFY_TOKEN and APIFY_ACTOR_ID.")

        run_input: Dict[str, Any] = {
            "query": query,
            "maxItems": limit,
        }
        if filters.category:
            run_input["category"] = filters.category
        if filters.price_min is not None:
            run_input["priceMin"] = filters.price_min
        if filters.price_max is not None:
            run_input["priceMax"] = filters.price_max
        if filters.year is not None:
            run_input["year"] = filters.year

        # Start actor run
        run_url = f"https://api.apify.com/v2/acts/{self.actor_id}/runs?token={self.token}"
        resp = requests.post(run_url, json=run_input, timeout=self.timeout_s)
        resp.raise_for_status()
        run = resp.json().get("data", {})
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            raise RuntimeError("Apify run did not return defaultDatasetId")

        items_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?clean=true&format=json&token={self.token}"
        items_resp = requests.get(items_url, timeout=self.timeout_s)
        items_resp.raise_for_status()
        items = items_resp.json() or []

        out: List[ProductRecord] = []
        for idx, item in enumerate(items):
            try:
                out.append(
                    ProductRecord(
                        id=str(item.get("id") or item.get("asin") or item.get("sku") or f"apify-{idx}"),
                        name=str(item.get("name") or item.get("title") or "").strip(),
                        price=_to_float(item.get("price")),
                        category=item.get("category"),
                        year=_to_int(item.get("year")),
                        specs=str(item.get("specs") or item.get("description") or "") or None,
                        image_url=item.get("image_url") or item.get("image") or item.get("imageUrl"),
                        source="apify",
                        source_url=item.get("url") or item.get("productUrl"),
                        updated_at=None,
                    )
                )
            except Exception:
                continue

        return [p for p in out if p.name]


def _to_float(v: Any) -> Optional[float]:  # pragma: no cover
    try:
        if v is None:
            return None
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None


def _to_int(v: Any) -> Optional[int]:  # pragma: no cover
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None

