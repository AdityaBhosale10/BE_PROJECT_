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
