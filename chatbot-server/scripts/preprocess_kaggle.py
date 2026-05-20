from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_price(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value)
    s = s.replace(",", "")
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    m = re.search(r"(19\d{2}|20\d{2})", str(value))
    if not m:
        return None
    y = int(m.group(1))
    if 1900 <= y <= 2100:
        return y
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Kaggle product dataset -> canonical JSONL")
    parser.add_argument("--input", required=True, help="Path to input CSV/JSON/Parquet")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--name-col", default="name", help="Column name for product name/title")
    parser.add_argument("--price-col", default="price", help="Column name for price")
    parser.add_argument("--category-col", default="category", help="Column name for category")
    parser.add_argument("--year-col", default="year", help="Column name for year")
    parser.add_argument("--specs-col", default="specs", help="Column name for specs/description")
    parser.add_argument("--image-col", default="image_url", help="Column name for image url")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(input_path, lines=input_path.suffix.lower() == ".jsonl")
    else:
        raise SystemExit(f"Unsupported input format: {input_path.suffix}")

    records = []
    for i, row in df.iterrows():
        name = _clean_text(row.get(args.name_col))
        if not name:
            continue
        rec: Dict[str, Any] = {
            "id": str(i),
            "name": name,
            "price": _parse_price(row.get(args.price_col)),
            "category": _clean_text(row.get(args.category_col)) or None,
            "year": _parse_year(row.get(args.year_col)),
            "specs": _clean_text(row.get(args.specs_col)) or None,
            "image_url": _clean_text(row.get(args.image_col)) or None,
            "source": "kaggle",
            "source_url": None,
            "updated_at": None,
        }
        records.append(rec)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()

