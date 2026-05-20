import json
import numpy as np

from src.repositories.faiss_repository import FaissRepository


def test_faiss_repository_save_load_and_search(tmp_path) -> None:
    repo = FaissRepository(str(tmp_path))

    products = [
        {"id": "0", "name": "alpha"},
        {"id": "1", "name": "bravo"},
        {"id": "2", "name": "charlie"},
    ]

    # Simple 3x2 vectors
    vec = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )
    index = repo.build_cosine_index(vec)
    repo.save(index, products)

    assert repo.has_index() is True

    loaded_index, loaded_products = repo.load()
    assert loaded_products == products

    # Query close to first vector
    q = np.asarray([1.0, 0.0], dtype=np.float32)
    scores, ids = repo.search_cosine(loaded_index, q, top_k=2)

    assert ids[0] == 0
    assert scores[0] >= scores[1]
