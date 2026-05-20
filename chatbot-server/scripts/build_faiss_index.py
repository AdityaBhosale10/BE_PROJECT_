from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.repositories.faiss_repository import FaissRepository
from src.services.embeddings import EmbeddingsService
from src.services.faiss_vector_store import FaissVectorStoreService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from processed products.jsonl")
    parser.add_argument("--products", required=True, help="Path to processed products.jsonl")
    parser.add_argument("--faiss-dir", required=True, help="Output directory for FAISS artifacts")
    parser.add_argument("--text-field", default="name", help="Field used for text embedding")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name",
    )
    args = parser.parse_args()

    products_path = Path(args.products)
    products = []
    with products_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            products.append(json.loads(line))

    repo = FaissRepository(args.faiss_dir)
    embeddings = EmbeddingsService(provider_type="sentence_transformers", model_name=args.model)
    svc = FaissVectorStoreService(repo=repo, embeddings=embeddings)
    svc.rebuild_from_products(products, text_field=args.text_field)

    print(f"Built FAISS index with {len(products)} products at {args.faiss_dir}")


if __name__ == "__main__":
    main()

