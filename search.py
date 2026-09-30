"""Step 3: search the index.

How a search works:
  1. Turn the user's sentence into a vector (same model as in index.py).
  2. Ask the database for the stored vectors closest to it.
  3. Return the GIFs those vectors belong to.

Try it:  python search.py a dog being confused
"""
import chromadb
from sentence_transformers import SentenceTransformer

from config import COLLECTION, DB_PATH, MODEL_NAME

# Loading the model is slow, so we do it once and keep it here for later searches.
_model = None
_collection = None


def _load():
    """Load the model and open the database (only the first time)."""
    global _model, _collection
    if _model is None:
        try:
            _collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)
        except Exception as e:
            raise RuntimeError("No GIF index found. Run `python index.py` first.") from e
        _model = SentenceTransformer(MODEL_NAME)
    return _model, _collection


def search_gif(query, k=10):
    """Find the k GIFs that best match `query`.

    Returns a list of dicts: {"url": ..., "description": ..., "score": ...}
    score is between 0 and 1. Higher = better match.
    """
    model, collection = _load()

    query_vector = model.encode(query).tolist()
    result = collection.query(query_embeddings=query_vector, n_results=k)

    # Chroma returns lists of lists (one inner list per query). We sent one query, so use [0].
    matches = []
    for meta, description, distance in zip(
        result["metadatas"][0], result["documents"][0], result["distances"][0]
    ):
        matches.append({
            "url": meta["url"],
            "description": description,
            "score": 1 - distance,  # Chroma gives distance (small = close). Flip it into a score.
        })
    return matches


if __name__ == "__main__":
    import sys

    # Use the words typed after "python search.py" as the query.
    query = " ".join(sys.argv[1:]) or "a dog being confused"
    for r in search_gif(query):
        print(f"{r['score']:.3f}  {r['description']}  {r['url']}")
