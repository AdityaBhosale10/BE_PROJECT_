from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.repositories.faiss_repository import FaissRepository
from src.services.clip_embeddings import ClipEmbeddingsService
from src.services.embeddings import EmbeddingsService
from src.services.multimodal_vector_store import MultimodalVectorStoreService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal FAISS indexes (text + CLIP image)")
    parser.add_argument(
        "--products",
        default=str(Path(__file__).resolve().parents[1] / "data" / "seed" / "products.jsonl"),
        help="Path to processed products.jsonl",
    )
    parser.add_argument(
        "--faiss-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "faiss"),
        help="Base output directory (text/ and image/ subdirs are created)",
    )
    parser.add_argument("--text-field", default="name", help="Field used for text embedding")
    parser.add_argument("--image-field", default="image_url", help="Field used for image URL embedding")
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

    text_repo = FaissRepository(str(Path(args.faiss_dir) / "text"))
    image_repo = FaissRepository(str(Path(args.faiss_dir) / "image"))
    text_embeddings = EmbeddingsService(provider_type="sentence_transformers", model_name=args.model)
    clip_embeddings = ClipEmbeddingsService()
    svc = MultimodalVectorStoreService(
        text_repo=text_repo,
        image_repo=image_repo,
        text_embeddings=text_embeddings,
        clip_embeddings=clip_embeddings,
        seed_path=str(products_path),
    )
    svc.rebuild(products, text_field=args.text_field, image_field=args.image_field)
    print(f"Built multimodal FAISS indexes with {len(products)} products at {args.faiss_dir}")


if __name__ == "__main__":
    main()
