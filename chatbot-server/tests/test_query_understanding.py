from src.services.query_understanding import QueryUnderstandingService


def test_extract_year_and_price_max() -> None:
    svc = QueryUnderstandingService()
    res = svc.extract("best smartphones under 45000 in 2026")
    assert res.filters.year == 2026
    assert res.filters.price_max == 45000


def test_extract_price_between() -> None:
    svc = QueryUnderstandingService()
    res = svc.extract("laptops between 50,000 and 90,000")
    assert res.filters.price_min == 50000
    assert res.filters.price_max == 90000


def test_extract_price_min_over() -> None:
    svc = QueryUnderstandingService()
    res = svc.extract("phones over 30000")
    assert res.filters.price_min == 30000


def test_llm_fallback_merge_prefers_deterministic() -> None:
    class FakeLLM:
        def generate(self, prompt: str, model: str) -> str:
            return (
                '{"category":"smartphone","brand":"x","year":2025,'
                '"price_min":10000,"price_max":99999,"other_specs":{"ram_gb":8}}'
            )

    svc = QueryUnderstandingService(llm_client=FakeLLM(), llm_model="m")
    # deterministic year + price_max should win over llm year/price_max
    res = svc.extract("smartphones under 45000 in 2026", use_llm_fallback=True)
    assert res.filters.year == 2026
    assert res.filters.price_max == 45000
    assert res.filters.category == "smartphone"
    assert res.filters.brand == "x"
    assert res.filters.other_specs.get("ram_gb") == 8


def test_llm_fallback_no_client_is_safe() -> None:
    svc = QueryUnderstandingService()
    res = svc.extract("cameras in 2024", use_llm_fallback=True)
    assert res.filters.year == 2024

