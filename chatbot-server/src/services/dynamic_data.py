from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.models.schemas import ProductRecord, QueryFilters

logger = logging.getLogger(__name__)


def needs_dynamic_data(query: str, filters: QueryFilters, *, current_year: Optional[int] = None) -> bool:
    """
    Heuristic gate for calling Apify.
    """
    q = (query or "").lower()
    if any(k in q for k in ["latest", "new launch", "newly launched", "just released", "2026", "2025"]):
        return True
    if filters.year is not None:
        now_year = current_year or datetime.utcnow().year
        if filters.year >= now_year:
            return True
    return False


def upsert_products_jsonl(path: str, products: Sequence[ProductRecord]) -> int:
    """
    Upsert by `id` into a JSONL file. Returns number of records written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    by_id: Dict[str, dict] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = str(obj.get("id") or "")
            if pid:
                by_id[pid] = obj

    for prod in products:
        by_id[prod.id] = prod.model_dump()

    with p.open("w", encoding="utf-8") as f:
        for _, obj in by_id.items():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return len(by_id)

