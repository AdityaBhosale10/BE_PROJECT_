from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.repositories.faiss_repository import FaissRepository
from src.services.embeddings import EmbeddingsService
from src.services.faiss_vector_store import FaissVectorStoreService
from src.services.evaluation import ndcg_at_k, precision_at_k, timed_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with Precision@K, nDCG, latency")
    parser.add_argument("--faiss-dir", required=True)
    parser.add_argument("--eval", required=True, help="JSONL with {query, relevant_ids:[...]}")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    repo = FaissRepository(args.faiss_dir)
    emb = EmbeddingsService(provider_type="sentence_transformers", model_name=args.model)
    store = FaissVectorStoreService(repo=repo, embeddings=emb)

    total_p = 0.0
    total_n = 0.0
    total_ms = 0.0
    n = 0

    for line in Path(args.eval).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        q = case["query"]
        rel = set(map(str, case.get("relevant_ids") or []))

        tr = timed_retrieval(lambda: [r["document"]["id"] for r in store.search(q, top_k=args.k)])
        p = precision_at_k(tr.retrieved_ids, rel, args.k)
        nd = ndcg_at_k(tr.retrieved_ids, rel, args.k)

        total_p += p
        total_n += nd
        total_ms += tr.elapsed_ms
        n += 1

    if n == 0:
        raise SystemExit("No eval cases found.")

    print(f"cases={n}")
    print(f"precision@{args.k}={total_p/n:.4f}")
    print(f"ndcg@{args.k}={total_n/n:.4f}")
    print(f"avg_latency_ms={total_ms/n:.2f}")


if __name__ == "__main__":
    main()

