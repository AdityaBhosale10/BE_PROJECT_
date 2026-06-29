import numpy as np

from src.repositories.faiss_repository import FaissRepository
from src.services.multimodal_vector_store import MultimodalVectorStoreService


class TinyTextEmbeddings:
    def embed_text(self, text: str):
        return [float(len(text)), 0.0]

    def embed_texts(self, texts):
        return [self.embed_text(t) for t in texts]


class TinyClip:
    def embed_text(self, text: str):
        # Different space than text embeddings but same dim
        return [0.0, float(len(text))]

    def embed_image_url(self, url: str):
        return [0.0, float(len(url))]

    def embed_image_bytes(self, content: bytes):
        return [0.0, float(len(content))]


def test_multimodal_search_fuses_results(tmp_path) -> None:
    text_repo = FaissRepository(str(tmp_path / "text"))
    image_repo = FaissRepository(str(tmp_path / "image"), index_filename="image.index", meta_filename="products.jsonl")

    svc = MultimodalVectorStoreService(
        text_repo=text_repo,
        image_repo=image_repo,
        text_embeddings=TinyTextEmbeddings(),
        clip_embeddings=TinyClip(),
    )

    products = [
        {"id": "0", "name": "alpha", "image_url": "http://x/a.jpg"},
        {"id": "1", "name": "beta", "image_url": "http://x/b.jpg"},
        {"id": "2", "name": "gamma", "image_url": ""},
    ]
    svc.rebuild(products)

    res = svc.search("alpha", top_k=2)
    assert len(res) == 2
    assert "document" in res[0]
    assert "image_url" in res[0]["document"]


def test_ensure_indexes_builds_from_seed(tmp_path) -> None:
    text_repo = FaissRepository(str(tmp_path / "text"))
    image_repo = FaissRepository(str(tmp_path / "image"))

    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(
        '{"id": "0", "name": "alpha saree", "description": "silk", "image_url": "http://x/a.jpg"}\n',
        encoding="utf-8",
    )

    svc = MultimodalVectorStoreService(
        text_repo=text_repo,
        image_repo=image_repo,
        text_embeddings=TinyTextEmbeddings(),
        clip_embeddings=TinyClip(),
        seed_path=str(seed_path),
    )

    assert svc.ensure_indexes() is True
    assert text_repo.has_index()
    assert image_repo.has_index()

    res = svc.search("alpha", top_k=1)
    assert len(res) == 1
    assert res[0]["document"]["name"] == "alpha saree"


def test_add_image_indexes_single_item(tmp_path) -> None:
    text_repo = FaissRepository(str(tmp_path / "text"))
    image_repo = FaissRepository(str(tmp_path / "image"), index_filename="image.index", meta_filename="products.jsonl")

    svc = MultimodalVectorStoreService(
        text_repo=text_repo,
        image_repo=image_repo,
        text_embeddings=TinyTextEmbeddings(),
        clip_embeddings=TinyClip(),
    )

    products = [
        {"id": "0", "name": "alpha", "image_url": "http://x/a.jpg"},
    ]
    svc.rebuild(products)

    idx = svc.add_image({"id": "1", "name": "beta", "image_url": ""}, image_bytes=b"12345")
    assert isinstance(idx, int)

    res = svc.search("", top_k=5, image_query_bytes=b"123")
    assert len(res) >= 1
