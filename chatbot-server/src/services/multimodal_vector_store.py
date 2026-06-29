from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.repositories.faiss_repository import FaissRepository
from src.services.clip_embeddings import ClipEmbeddingsService
from src.services.embeddings import EmbeddingsService

logger = logging.getLogger(__name__)

DEFAULT_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "seed" / "products.jsonl"


class MultimodalVectorStoreService:
    """
    Multimodal retrieval over:
    - text embeddings (Sentence-Transformers) in FAISS
    - image embeddings (CLIP) in FAISS

    For now, we keep separate indexes and fuse results by weighted score.
    """

    def __init__(
        self,
        text_repo: FaissRepository,
        image_repo: FaissRepository,
        text_embeddings: EmbeddingsService,
        clip_embeddings: ClipEmbeddingsService,
        *,
        seed_path: Optional[str] = None,
    ) -> None:
        self.text_repo = text_repo
        self.image_repo = image_repo
        self.text_embeddings = text_embeddings
        self.clip_embeddings = clip_embeddings
        self.seed_path = Path(seed_path) if seed_path else DEFAULT_SEED_PATH

        self._text_index = None
        self._image_index = None
        self._products: Optional[List[dict]] = None

    @staticmethod
    def _load_products_jsonl(path: Path) -> List[dict]:
        if not path.exists():
            return []
        products: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                products.append(json.loads(line))
        return products

    def _embed_product_for_index(
        self,
        product: dict,
        *,
        text_field: str = "name",
    ) -> List[float]:
        """
        Build a CLIP vector for catalog indexing.

        Uses rich text (name + description) so uploaded images can match via
        CLIP cross-modal similarity without fetching remote image URLs at build time.
        """
        name = str(product.get(text_field) or "")
        desc = str(product.get("description") or "")
        label = f"{name}. {desc}".strip(". ").strip()
        return self.clip_embeddings.embed_text(label or name)

    def ensure_indexes(self, products: Optional[Sequence[dict]] = None) -> bool:
        """
        Build missing FAISS indexes from seed/custom products.

        Returns True when both text and image indexes are available afterward.
        """
        if self.text_repo.has_index() and self.image_repo.has_index():
            return True

        catalog = list(products) if products is not None else self._load_products_jsonl(self.seed_path)
        if not catalog:
            logger.warning(
                "Multimodal FAISS indexes missing and no seed catalog found at %s",
                self.seed_path,
            )
            return False

        logger.info("Building multimodal FAISS indexes from %d seed products", len(catalog))
        self.rebuild(catalog)
        return self.text_repo.has_index() and self.image_repo.has_index()

    def _ensure_loaded(self) -> None:
        if self._products is not None and self._text_index is not None and self._image_index is not None:
            return

        if not self.text_repo.has_index() or not self.image_repo.has_index():
            self.ensure_indexes()

        if not self.text_repo.has_index() or not self.image_repo.has_index():
            self._products = []
            self._text_index = None
            self._image_index = None
            return

        self._text_index, self._products = self.text_repo.load()
        self._image_index, image_products = self.image_repo.load()
        if not self._products and image_products:
            self._products = image_products

    def rebuild(self, products: Sequence[dict], text_field: str = "name", image_field: str = "image_url") -> None:
        if not products:
            raise ValueError("Cannot rebuild multimodal indexes from an empty product list")

        # Text index
        texts = [str(p.get(text_field) or "") for p in products]
        tvec = np.asarray(self.text_embeddings.embed_texts(texts), dtype=np.float32)
        text_index = self.text_repo.build_cosine_index(tvec)
        self.text_repo.save(text_index, list(products))

        # Image index: CLIP text embeddings for cross-modal matching with user uploads
        ivec_list = [self._embed_product_for_index(p, text_field=text_field) for p in products]
        ivec = np.asarray(ivec_list, dtype=np.float32)
        image_index = self.image_repo.build_cosine_index(ivec)
        self.image_repo.save(image_index, list(products))

        self._text_index = text_index
        self._image_index = image_index
        self._products = list(products)
        logger.info("Rebuilt multimodal indexes with %d products", len(products))

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        image_query_url: Optional[str] = None,
        image_query_bytes: Optional[bytes] = None,
        text_weight: float = 0.7,
        image_weight: float = 0.3,
        allowed_ids: Optional[set[int]] = None,
    ) -> List[Dict]:
        self._ensure_loaded()
        if not self._products or self._text_index is None or self._image_index is None:
            logger.warning("Multimodal search skipped: FAISS indexes are not available")
            return []

        # Text search
        qtext = np.asarray(self.text_embeddings.embed_text(query), dtype=np.float32)
        tscores, tids = self.text_repo.search_cosine(self._text_index, qtext, top_k=top_k * 5)

        # Image search (optional)
        if image_query_bytes is not None:
            qimg = np.asarray(self.clip_embeddings.embed_image_bytes(image_query_bytes), dtype=np.float32)
        elif image_query_url:
            qimg = np.asarray(self.clip_embeddings.embed_image_url(image_query_url), dtype=np.float32)
        else:
            qimg = np.asarray(self.clip_embeddings.embed_text(query), dtype=np.float32)
        iscores, iids = self.image_repo.search_cosine(self._image_index, qimg, top_k=top_k * 5)

        fused: Dict[int, float] = {}
        for score, idx in zip(tscores.tolist(), tids.tolist()):
            if idx < 0:
                continue
            fused[idx] = fused.get(idx, 0.0) + text_weight * float(score)
        for score, idx in zip(iscores.tolist(), iids.tolist()):
            if idx < 0:
                continue
            fused[idx] = fused.get(idx, 0.0) + image_weight * float(score)

        # Rank and emit
        out: List[Dict] = []
        for idx, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            if allowed_ids is not None and idx not in allowed_ids:
                continue
            if idx >= len(self._products):
                continue
            out.append({"similarityScore": float(score), "document": self._products[idx]})
            if len(out) >= top_k:
                break
        return out

    def add_image(self, product: dict, image_bytes: bytes | None = None, image_url: str | None = None) -> int:
        """Index a single product with an image. Returns the assigned integer id."""
        self._ensure_loaded()

        if self._image_index is None or self._products is None:
            self._products = []
            self._image_index = None

        # compute vector
        if image_bytes is not None:
            vec = np.asarray(self.clip_embeddings.embed_image_bytes(image_bytes), dtype=np.float32)
        elif image_url:
            vec = np.asarray(self.clip_embeddings.embed_image_url(image_url), dtype=np.float32)
        else:
            vec = np.asarray(self.clip_embeddings.embed_text(str(product.get("name") or "")), dtype=np.float32)

        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        if self._image_index is None:
            self._image_index = self.image_repo.build_cosine_index(vec)
        else:
            try:
                self._image_index.add(vec)
            except Exception:
                prev_vecs = [self._embed_product_for_index(p) for p in self._products]
                prev_vecs.append(vec.flatten().tolist())
                arr = np.asarray(prev_vecs, dtype=np.float32)
                self._image_index = self.image_repo.build_cosine_index(arr)

        self._products.append(product)
        self.image_repo.save(self._image_index, list(self._products))

        if self._text_index is None and self.text_repo.has_index():
            self._text_index, _ = self.text_repo.load()

        return len(self._products) - 1
