"""Step 2: build the search index (run this once).

Idea: a computer cannot compare sentences directly, but it can compare lists of numbers.
An "embedding model" turns a sentence into a list of 384 numbers (a vector).
Sentences with similar meaning get similar vectors.
We turn every GIF description into a vector and save them in a vector database (Chroma).

Usage:  python index.py --limit 5000
"""
import argparse
import os

import chromadb
from sentence_transformers import SentenceTransformer

from data import load_data

# Settings shared with search.py
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast model: 384 numbers per sentence
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")  # database folder
COLLECTION = "gif-search"  # a "collection" is like a table in the database
BATCH_SIZE = 64  # encode this many descriptions at a time (faster than one by one)


def main():
    # Read the optional --limit option from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000, help="how many GIFs to index")
    args = parser.parse_args()

    # Open (or create) the database. "cosine" = compare vectors by the angle between them.
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    # Do not index twice. To start over, delete the chroma_db folder.
    if collection.count() > 0:
        print(f"Index already has {collection.count()} GIFs. Delete '{DB_PATH}' to rebuild.")
        return

    df = load_data(limit=args.limit)
    model = SentenceTransformer(MODEL_NAME)  # downloads the model the first time (~90 MB)

    # Go through the data in small batches.
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        descriptions = batch["description"].tolist()

        vectors = model.encode(descriptions).tolist()  # sentences -> numbers

        collection.add(
            ids=[str(i) for i in batch.index],  # every item needs a unique id
            embeddings=vectors,  # what we search on
            documents=descriptions,  # the original text, to show in results
            metadatas=[{"url": u} for u in batch["url"]],  # extra info: the GIF link
        )
        print(f"Indexed {min(start + BATCH_SIZE, len(df))}/{len(df)}", end="\r")

    print(f"\nDone. {collection.count()} GIFs indexed.")


if __name__ == "__main__":
    main()
