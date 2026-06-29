from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import vector_search as vs


def test_multimodal_search_endpoint_returns_501_when_not_configured():
    app = FastAPI()
    app.include_router(vs.router)
    client = TestClient(app)

    res = client.post("/api/vector/multimodal/search", data={"query": "x"})
    assert res.status_code == 501


def test_multimodal_search_with_fake_service_file_upload():
    app = FastAPI()
    app.include_router(vs.router)

    class Fake:
        def search(self, query, top_k=5, image_query_url=None, image_query_bytes=None):
            return [{"similarityScore": 0.9, "document": {"id": "1", "name": "img", "image_url": "http://x/a.jpg"}}]

    app.state.multimodal_service = Fake()
    client = TestClient(app)

    files = {"image": ("test.jpg", b"123", "image/jpeg")}
    data = {"query": "abc", "top_k": "3"}
    res = client.post("/api/vector/multimodal/search", data=data, files=files)
    assert res.status_code == 200
    j = res.json()
    assert "results" in j and len(j["results"]) == 1


def test_image_upload_indexes_via_fake_service():
    app = FastAPI()
    app.include_router(vs.router)

    class Fake:
        def add_image(self, product, image_bytes=None, image_url=None):
            return 5

    app.state.multimodal_service = Fake()
    client = TestClient(app)

    files = {"image": ("a.jpg", b"abc", "image/jpeg")}
    data = {"title": "My Photo"}
    res = client.post("/api/vector/images/upload", data=data, files=files)
    assert res.status_code == 200
    assert res.json().get("ok") is True
