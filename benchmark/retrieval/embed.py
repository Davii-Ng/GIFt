"""Dataset 1 / step 1: create or reuse embeddings for ONE chosen model.

Run from GIFt-main (new package):
  python -m benchmark.retrieval.embed --model qwen3 --limit 5
  python -m benchmark.retrieval.embed --model qwen3 --full
  python -m benchmark.retrieval.embed --model bge-m3 --full
  python -m benchmark.retrieval.embed --model gemini --full --kind queries

Single frozen Dataset 1: 10,000 TGIF captions, 200 held-out queries.
Corpus/query embeddings are cached together in data/benchmark/embeddings/dataset1.
Without --full, only five query rows are encoded as a quick smoke test.
Existing valid vectors and finished chunks are reused. No file is deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmark"
DATASET = DATA / "dataset1"
OUTPUT = DATA / "embeddings" / "dataset1"
V2_DATASET = DATASET
V2_OUTPUT = OUTPUT
MODEL_IDS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "gemini": "gemini-embedding-001",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "bge-m3": "BAAI/bge-m3",
}
QWEN_QUERY_INSTRUCTION = (
    "Instruct: Retrieve GIF captions that match the user's search query.\nQuery: "
)


def normalize(vectors):
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Embeddings must be a finite 2D array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding contains a zero vector")
    return x / norms


class LocalEmbedder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.name = name
        print(f"Loading {MODEL_IDS[name]} on CPU...")
        self.model = SentenceTransformer(
            MODEL_IDS[name], device="cpu", trust_remote_code=False
        )

    def encode(self, texts, role, batch_size):
        if self.name == "qwen3" and role == "query":
            texts = [QWEN_QUERY_INSTRUCTION + text for text in texts]
        options = dict(
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        if self.name == "qwen3":
            options["prompt"] = ""
        return normalize(self.model.encode(texts, **options))


class GeminiEmbedder:
    def __init__(self, min_delay=1.5):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=api_key)
        self.last_call = None
        self.min_delay = float(min_delay)

    def encode(self, texts, role, batch_size):
        from google.genai import errors, types
        task = "RETRIEVAL_QUERY" if role == "query" else "RETRIEVAL_DOCUMENT"
        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            for attempt in range(4):
                if self.last_call is not None:
                    delay = self.min_delay - (time.monotonic() - self.last_call)
                    if delay > 0:
                        time.sleep(delay)
                self.last_call = time.monotonic()
                try:
                    response = self.client.models.embed_content(
                        model=MODEL_IDS["gemini"],
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task),
                    )
                    break
                except errors.APIError as exc:
                    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                    # Daily free-tier quota will not be fixed by repeated retries.
                    if "PerDay" in str(exc) or "per day" in str(exc).lower():
                        raise RuntimeError(
                            "Gemini daily quota exhausted. Completed chunks are saved; "
                            "run the same command again once quota resets."
                        ) from exc
                    if code not in (429, 500, 502, 503, 504) or attempt == 3:
                        raise
                    delay = min(60, 10 * (2 ** attempt) + random.random())
                    print(f"Gemini HTTP {code}; retrying after {delay:.0f}s...")
                    time.sleep(delay)
            if not response.embeddings or len(response.embeddings) != len(batch):
                raise RuntimeError("Gemini returned the wrong number of vectors")
            results.extend(e.values for e in response.embeddings)
        return normalize(results)


def read_texts(path, field, limit):
    if not path.is_file():
        raise FileNotFoundError(f"Missing Dataset 1 file: {path}")
    texts = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                value = json.loads(line)[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Empty {field} in {path.name}")
                texts.append(value)
                if limit is not None and len(texts) >= limit:
                    break
    if not texts:
        raise ValueError(f"No texts in {path}")
    return texts


def fingerprint(texts, model, kind):
    # Keep the original fingerprint layout so old saved chunks remain compatible.
    settings = {
        "model": model,
        "model_id": MODEL_IDS[model],
        "kind": kind,
        "qwen_query_instruction": QWEN_QUERY_INSTRUCTION if model == "qwen3" else None,
        "gemini_task": (
            "RETRIEVAL_QUERY" if kind == "queries" else "RETRIEVAL_DOCUMENT"
        ) if model == "gemini" else None,
        "texts": texts,
    }
    raw = json.dumps(settings, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def embed_kind(model, kind, full, limit, batch_size, chunk_size, dataset="v2", gemini_delay=1.5):
    dataset_dir = DATASET
    output_dir = OUTPUT
    filename = "corpus.jsonl" if kind == "corpus" else "queries.jsonl"
    field = "caption" if kind == "corpus" else "query"
    role = "document" if kind == "corpus" else "query"
    texts = read_texts(dataset_dir / filename, field, None if full else limit)
    mode = "full" if full else f"sample_{len(texts)}"
    folder = output_dir / model / mode / kind
    folder.mkdir(parents=True, exist_ok=True)
    vectors_file = folder / "vectors.npy"
    metadata_file = folder / "metadata.json"
    reuse_file = folder / "reuse_metadata.json"
    signature = fingerprint(texts, model, kind)
    metadata = {
        "model": model,
        "model_id": MODEL_IDS[model],
        "dataset": "dataset1_v2" if dataset == "v2" else "dataset1",
        "kind": kind,
        "mode": mode,
        "count": len(texts),
        "chunk_size": chunk_size,
        "fingerprint": signature,
        "normalized": True,
        "dtype": "float32",
    }

    if metadata_file.is_file():
        old = json.loads(metadata_file.read_text(encoding="utf-8"))
        for key in ("model_id", "kind", "mode", "count", "fingerprint", "chunk_size"):
            if old.get(key) != metadata[key]:
                raise RuntimeError(
                    f"Cache settings changed: {metadata_file} ({key}). "
                    "Don't overwrite it; inspect the previous embeddings first."
                )

    if vectors_file.is_file():
        # The reused MiniLM vectors have reuse_metadata.json instead of metadata.json.
        if not metadata_file.is_file():
            if not reuse_file.is_file():
                raise RuntimeError(f"Existing vectors lack verification metadata: {vectors_file}")
            reused = json.loads(reuse_file.read_text(encoding="utf-8"))
            if (reused.get("count") != len(texts) or reused.get("kind") != kind
                    or reused.get("model") != model or reused.get("reused_from_original") is not True):
                raise RuntimeError(f"Reuse metadata does not match: {reuse_file}")
        arr = np.load(vectors_file, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.dtype != np.float32:
            raise RuntimeError(f"Existing vectors have unexpected shape/dtype: {vectors_file}")
        print(f"{model} / {kind}: already complete ({arr.shape}); skipping.")
        return

    if not metadata_file.is_file():
        write_json(metadata_file, metadata)

    chunks = folder / "chunks"
    chunks.mkdir(exist_ok=True)
    ranges = [(s, min(s + chunk_size, len(texts))) for s in range(0, len(texts), chunk_size)]
    missing = [(s, e) for s, e in ranges
               if not (chunks / f"{s:07d}_{e:07d}.npy").is_file()]
    print(f"{model} / {kind}: {len(texts)} texts, {len(missing)} missing chunks")
    embedder = None
    if missing:
        embedder = GeminiEmbedder(min_delay=gemini_delay) if model == "gemini" else LocalEmbedder(model)
    for s, e in missing:
        vectors = embedder.encode(texts[s:e], role=role, batch_size=batch_size)
        if vectors.shape[0] != e - s:
            raise RuntimeError(f"Expected {e - s} vectors, got {vectors.shape}")
        filename = f"{s:07d}_{e:07d}.npy"
        tmp = chunks / (filename + ".partial.npy")
        np.save(tmp, vectors.astype(np.float32, copy=False))
        tmp.replace(chunks / filename)
        print(f"Saved {kind} {s}:{e} / {len(texts)}", flush=True)

    first = np.load(chunks / f"{ranges[0][0]:07d}_{ranges[0][1]:07d}.npy", mmap_mode="r")
    dim = first.shape[1]
    temporary = folder / "vectors.partial.npy"
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                       shape=(len(texts), dim))
    for s, e in ranges:
        chunk = np.load(chunks / f"{s:07d}_{e:07d}.npy", mmap_mode="r")
        if chunk.shape != (e-s, dim) or not np.isfinite(chunk).all():
            raise RuntimeError(f"Bad chunk for {model} {kind} rows {s}:{e}")
        matrix[s:e] = chunk
    matrix.flush()
    del matrix
    temporary.replace(vectors_file)
    metadata["dimension"] = dim
    write_json(metadata_file, metadata)
    print(f"DONE: {vectors_file} ({len(texts)}, {dim})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--dataset", choices=("v2",), default="v2",
                        help="Single frozen 10k/200 corpus in data/benchmark/dataset1")
    parser.add_argument("--kind", choices=("both", "corpus", "queries"), default="both")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--full", action="store_true", help="Embed all 10K captions + 200 queries")
    scope.add_argument("--limit", type=int, default=5, help="Test on first N rows (default: 5)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--gemini-delay", type=float, default=1.5,
                        help="Minimum seconds between Gemini API requests (e.g. 6). Does not bypass daily quotas")
    args = parser.parse_args()
    if args.limit < 1 or args.chunk_size < 1 or args.gemini_delay < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("--limit, --batch-size and --chunk-size must be positive; --gemini-delay cannot be negative")
    batch = args.batch_size or {"minilm": 8, "gemini": 2, "qwen3": 2, "bge-m3": 2}[args.model]
    kinds = ("corpus", "queries") if args.kind == "both" else (args.kind,)
    if args.model == "gemini":
        print(f"Gemini rate limit: at least {args.gemini_delay:g}s between requests; batch-size={batch}.")
        print("Note: delay cannot prevent daily quota exhaustion or paid API charges.")
    for kind in kinds:
        embed_kind(args.model, kind, args.full, args.limit, batch, args.chunk_size, args.dataset, args.gemini_delay)


if __name__ == "__main__":
    main()
