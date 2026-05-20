import json

from src.models.schemas import ProductRecord, QueryFilters
from src.services.dynamic_data import needs_dynamic_data, upsert_products_jsonl


def test_needs_dynamic_data_latest_keyword() -> None:
    assert needs_dynamic_data("latest smartphones", QueryFilters()) is True


def test_needs_dynamic_data_future_year() -> None:
    assert needs_dynamic_data("phones", QueryFilters(year=2026), current_year=2025) is True


def test_needs_dynamic_data_past_year_false() -> None:
    assert needs_dynamic_data("phones", QueryFilters(year=2022), current_year=2025) is False


def test_upsert_products_jsonl(tmp_path) -> None:
    path = tmp_path / "products.jsonl"
    p1 = ProductRecord(id="1", name="a", source="apify")
    p2 = ProductRecord(id="2", name="b", source="apify")
    count = upsert_products_jsonl(str(path), [p1, p2])
    assert count == 2

    # Update existing id=2
    p2b = ProductRecord(id="2", name="b2", source="apify")
    count2 = upsert_products_jsonl(str(path), [p2b])
    assert count2 == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    ids = {json.loads(l)["id"]: json.loads(l)["name"] for l in lines if l.strip()}
    assert ids["2"] == "b2"

