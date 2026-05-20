from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from src.interfaces.base import LLMClientInterface
from src.models.schemas import QueryFilters

logger = logging.getLogger(__name__)


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_PRICE_UNDER_RE = re.compile(r"\b(under|below|less than)\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+)\b", re.I)
_PRICE_OVER_RE = re.compile(r"\b(over|above|more than)\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+)\b", re.I)
_PRICE_BETWEEN_RE = re.compile(
    r"\b(between)\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+)\s*(?:and|to|-)\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+)\b",
    re.I,
)


def _to_number(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").strip())
    except Exception:
        return None


@dataclass(frozen=True)
class QueryUnderstandingResult:
    semantic_query: str
    filters: QueryFilters


class QueryUnderstandingService:
    """
    Extract structured filters from a user query.

    Stage-1 implementation:
    - deterministic regex extraction for year + price ranges
    - optional LLM fallback for category/brand/specs (kept off by default)
    """

    def __init__(self, llm_client: Optional[LLMClientInterface] = None, llm_model: str = "") -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model

    def extract(self, query: str, use_llm_fallback: bool = False) -> QueryUnderstandingResult:
        filters = QueryFilters()

        # Year
        m = _YEAR_RE.search(query)
        if m:
            filters.year = int(m.group(1))

        # Price
        m = _PRICE_BETWEEN_RE.search(query)
        if m:
            lo = _to_number(m.group(2))
            hi = _to_number(m.group(3))
            if lo is not None:
                filters.price_min = lo
            if hi is not None:
                filters.price_max = hi
        else:
            m = _PRICE_UNDER_RE.search(query)
            if m:
                hi = _to_number(m.group(2))
                if hi is not None:
                    filters.price_max = hi
            m = _PRICE_OVER_RE.search(query)
            if m:
                lo = _to_number(m.group(2))
                if lo is not None:
                    filters.price_min = lo

        semantic_query = query.strip()

        if use_llm_fallback and self.llm_client and self.llm_model:
            try:
                llm_filters = self._llm_extract(query)
                filters = self._merge(filters, llm_filters)
            except Exception as e:
                logger.warning("LLM filter extraction failed: %s", e)

        return QueryUnderstandingResult(semantic_query=semantic_query, filters=filters)

    def _llm_extract(self, query: str) -> QueryFilters:
        prompt = (
            "Extract structured filters from the user query and return ONLY valid JSON for this schema:\n"
            "{category: string|null, brand: string|null, year: number|null, price_min: number|null, price_max: number|null, other_specs: object}\n"
            f"User query: {query}\n"
        )
        raw = self.llm_client.generate(prompt=prompt, model=self.llm_model)
        data = json.loads(raw)
        return QueryFilters(**data)

    @staticmethod
    def _merge(base: QueryFilters, extra: QueryFilters) -> QueryFilters:
        # Prefer deterministic values when present
        merged = base.model_copy(deep=True)
        for field in ("category", "brand", "year", "price_min", "price_max"):
            if getattr(merged, field) is None and getattr(extra, field) is not None:
                setattr(merged, field, getattr(extra, field))
        merged.other_specs = {**(extra.other_specs or {}), **(merged.other_specs or {})}
        return merged

