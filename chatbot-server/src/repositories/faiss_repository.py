from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss  # type: ignore
import numpy as np


@dataclass(frozen=True)
class FaissPaths:
    base_dir: Path
    index_filename: str = "faiss.index"
    meta_filename: str = "products.jsonl"

    @property
    def index_path(self) -> Path:
        return self.base_dir / self.index_filename

    @property
    def meta_path(self) -> Path:
        return self.base_dir / self.meta_filename


class FaissRepository:
    """
    Minimal FAISS persistence layer:
    - Stores the FAISS index to disk
    - Stores product metadata as JSONL aligned by internal integer ID
    """

    def __init__(self, base_dir: str, *, index_filename: str = "faiss.index", meta_filename: str = "products.jsonl") -> None:
        self.paths = FaissPaths(base_dir=Path(base_dir), index_filename=index_filename, meta_filename=meta_filename)
        self.paths.base_dir.mkdir(parents=True, exist_ok=True)

    def has_index(self) -> bool:
        return self.paths.index_path.exists() and self.paths.meta_path.exists()

    def save(self, index: faiss.Index, products: Sequence[dict]) -> None:
        faiss.write_index(index, str(self.paths.index_path))
        with self.paths.meta_path.open("w", encoding="utf-8") as f:
            for p in products:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def load(self) -> Tuple[faiss.Index, List[dict]]:
        index = faiss.read_index(str(self.paths.index_path))
        products: List[dict] = []
        with self.paths.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                products.append(json.loads(line))
        return index, products

    @staticmethod
    def build_cosine_index(vectors: np.ndarray) -> faiss.Index:
        """
        Build an inner-product index with normalized vectors => cosine similarity.

        vectors: float32 array of shape (N, D)
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        d = vectors.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(vectors)
        return index

    @staticmethod
    def search_cosine(
        index: faiss.Index,
        query_vector: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (scores, ids) arrays.
        """
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        scores, ids = index.search(query_vector, top_k)
        return scores[0], ids[0]
