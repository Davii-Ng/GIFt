
"""Step 3: Search GIFs using SentenceTransformer and ChromaDB."""

import time
import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION, DB_PATH, MODEL_NAME

# Cache model and collection
_model = None
_collection = None


def _load():
    """Load model and database only once."""
    global _model, _collection

    if _model is None:
        try:
            _collection = chromadb.PersistentClient(
                path=DB_PATH
            ).get_collection(COLLECTION)
        except Exception as e:
            raise RuntimeError(
                "No GIF index found. Run `python index.py` first."
            ) from e

        _model = SentenceTransformer(MODEL_NAME)

    return _model, _collection


def search_gif(query, k=10, return_timing=False):
    """Search for the top-K most similar GIFs."""

    total_start = time.perf_counter()

    # Step 1: Load model and database
    load_start = time.perf_counter()

    model, collection = _load()

    load_time = time.perf_counter() - load_start

    # Step 2: Convert query into embedding
    embedding_start = time.perf_counter()

    query_vector = model.encode(query).tolist()

    embedding_time = time.perf_counter() - embedding_start

    # Step 3: Search nearest neighbors in ChromaDB
    search_start = time.perf_counter()

    result = collection.query(
        query_embeddings=query_vector,
        n_results=k
    )

    vector_search_time = time.perf_counter() - search_start

    # Step 4: Format GIF results
    matches = []

    for meta, description, distance in zip(
        result["metadatas"][0],
        result["documents"][0],
        result["distances"][0]
    ):
        matches.append({
            "url": meta["url"],
            "description": description,
            "score": 1 - distance
        })

    total_time = time.perf_counter() - total_start

    # Convert seconds to milliseconds
    timings = {
        "load_ms": load_time * 1000,
        "embedding_ms": embedding_time * 1000,
        "vector_search_ms": vector_search_time * 1000,
        "total_ms": total_time * 1000
    }

    if return_timing:
        return matches, timings

    return matches


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "a dog being confused"

    results, timings = search_gif(
        query,
        return_timing=True
    )

    for r in results:
        print(
            f"{r['score']:.3f}  "
            f"{r['description']}  "
            f"{r['url']}"
        )

    print("\n--- Search Performance ---")
    print(f"Model Load: {timings['load_ms']:.2f} ms")
    print(f"Embedding: {timings['embedding_ms']:.2f} ms")
    print(f"Vector Search: {timings['vector_search_ms']:.2f} ms")
    print(f"Total Time: {timings['total_ms']:.2f} ms")
