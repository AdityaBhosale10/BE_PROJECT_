from src.repositories.faiss_repository import FaissRepository
from src.services.faiss_vector_store import FaissVectorStoreService


class TinyEmbeddings:
    def embed_text(self, text: str):
        # 2D deterministic embedding: (len, vowel_count)
        vowels = sum(1 for c in text.lower() if c in "aeiou")
        return [float(len(text)), float(vowels)]

    def embed_texts(self, texts):
        return [self.embed_text(t) for t in texts]


def test_faiss_vector_store_rebuild_and_search(tmp_path) -> None:
    repo = FaissRepository(str(tmp_path))
    svc = FaissVectorStoreService(repo=repo, embeddings=TinyEmbeddings())

    products = [
        {"id": "0", "name": "iphone 15", "price": 70000},
        {"id": "1", "name": "samsung galaxy", "price": 65000},
        {"id": "2", "name": "budget phone", "price": 15000},
    ]
    svc.rebuild_from_products(products, text_field="name")

    results = svc.search("iphone", top_k=2)
    assert len(results) == 2
    assert "document" in results[0]
    assert "similarityScore" in results[0]
