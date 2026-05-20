from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.repositories.faiss_repository import FaissRepository
from src.services.embeddings import EmbeddingsService

logger = logging.getLogger(__name__)


class FaissVectorStoreService:
    """
    Text-only FAISS vector store service (stage 1).

    This provides retrieval over a local FAISS index and product metadata stored as JSONL.
    """

    def __init__(self, repo: FaissRepository, embeddings: EmbeddingsService):
        self.repo = repo
        self.embeddings = embeddings
        self._index = None
        self._products: Optional[List[dict]] = None

    def _ensure_loaded(self) -> None:
        if self._index is None or self._products is None:
            if not self.repo.has_index():
                raise RuntimeError(
                    "FAISS index not found. Build it first (see scripts/preprocess_kaggle.py output + index builder)."
                )
            self._index, self._products = self.repo.load()

    def rebuild_from_products(self, products: Sequence[dict], text_field: str = "name") -> None:
        """
        Build a fresh index from the provided product records and persist it.
        """
        texts = [str(p.get(text_field) or "") for p in products]
        vectors = self.embeddings.embed_texts(texts)
        vec = np.asarray(vectors, dtype=np.float32)
        index = self.repo.build_cosine_index(vec)
        self.repo.save(index, list(products))
        self._index = index
        self._products = list(products)
        logger.info("FAISS index rebuilt with %d products", len(products))

    def search(self, query: str, top_k: int = 5, allowed_ids: Optional[set[int]] = None) -> List[Dict]:
        self._ensure_loaded()
        assert self._products is not None
        assert self._index is not None

        qvec = np.asarray(self.embeddings.embed_text(query), dtype=np.float32)
        scores, ids = self.repo.search_cosine(self._index, qvec, top_k=top_k * 5 if allowed_ids else top_k)

        results: List[Dict] = []
        for score, idx in zip(scores.tolist(), ids.tolist()):
            if idx < 0:
                continue
            if allowed_ids is not None and idx not in allowed_ids:
                continue
            if idx >= len(self._products):
                continue
            results.append({"similarityScore": float(score), "document": self._products[idx]})
            if len(results) >= top_k:
                break
        return results

