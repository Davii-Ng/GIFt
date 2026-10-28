"""Dataset 1 / V2 GIF retrieval benchmark, quality metrics and error analysis.

Default command (from the GIFt-main project root):
    python -m benchmark.retrieval.semantic_dataset1 --dataset v2 --model all

This command ONLY reads saved vectors. It does not call Gemini, download models,
re-embed captions, or rerun latency measurements. It writes:
    comparison.json                 summary for all models
    {model}_results.json            full query-by-query rankings
    benchmark_summary.csv           quality metrics + saved latency
    query_comparison.csv            ONE ROW per query, side-by-side model Top 5
    query_comparison_long.csv       one row per query/model/rank (filterable)
    query_comparison.html           visual GIF gallery for error analysis

After Gemini has both FULL cached corpus and query embeddings, run the same
command again. Gemini is automatically added to the comparison.

Optional explicit fresh CPU timing (200 queries by default):
    python -m benchmark.retrieval.semantic_dataset1 --dataset v2 --model all --measure-latency
Optional Gemini API timing (uses extra billable/quota-limited API requests!):
    python -m benchmark.retrieval.semantic_dataset1 --dataset v2 --model gemini --measure-latency --allow-gemini-api
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "benchmark"
DATASETS = {"v2": ROOT / "benchmark" / "datasets" / "dataset1"}
VECTOR_V1 = BASE / "embeddings" / "dataset1"
VECTOR_V2 = VECTOR_V1
MODEL_IDS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "bge-m3": "BAAI/bge-m3",
    "gemini": "gemini-embedding-001",
}
MODEL_NAMES = {
    "minilm": "MiniLM",
    "qwen3": "Qwen3",
    "bge-m3": "BGE-M3",
    "gemini": "Gemini",
}
ALL_MODELS = tuple(MODEL_IDS)
TOP_K = 5


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


def get_source(q: dict) -> str:
    source = str(q.get("source", "")).strip().lower().replace("-", "_")
    if source in {"a", "source_a"}:
        return "source_a"
    if source in {"b", "source_b"}:
        return "source_b"
    return source or "unknown"


def load_dataset(version: str) -> tuple[list[dict], list[dict]]:
    path = DATASETS[version]
    corpus = read_jsonl(path / "corpus.jsonl")
    queries = read_jsonl(path / "queries.jsonl")
    if not corpus or not queries:
        raise ValueError("Corpus and queries must be nonempty")
    doc_ids = [d["doc_id"] for d in corpus]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Corpus has duplicate doc_id values")
    qids = [q["query_id"] for q in queries]
    if len(qids) != len(set(qids)):
        raise ValueError("Queries have duplicate query_id values")
    corpus_gifs = {d["gif_id"] for d in corpus}
    for q in queries:
        relevant = q.get("relevant_ids")
        if not isinstance(relevant, list) or not relevant or not (set(relevant) & corpus_gifs):
            raise ValueError(f"Missing relevant GIF in corpus: {q['query_id']}")

    if version == "v2":
        # Old corpus vectors can ONLY be reused if corpus bytes are identical.
        corpus_texts = {normalized_text(d["caption"]) for d in corpus}
        doc_set = set(doc_ids)
        if len({normalized_text(q["query"]) for q in queries}) != len(queries):
            raise ValueError("V2 has duplicate query texts")
        if len(queries) != 200:
            raise ValueError(f"V2 should have 200 queries; found {len(queries)}")
        for q in queries:
            if get_source(q) != "source_a":
                raise ValueError(f"Not a Source A query: {q['query_id']}")
            if normalized_text(q["query"]) in corpus_texts:
                raise ValueError(f"Query text leaked into corpus: {q['query_id']}")
            if q.get("heldout_doc_id") in doc_set:
                raise ValueError(f"Held-out caption leaked into corpus: {q['query_id']}")
            if len(q["relevant_ids"]) != 1:
                raise ValueError(f"V2 should have one labeled GIF per query: {q['query_id']}")
    return corpus, queries


def vector_paths(model: str, version: str) -> tuple[Path, Path]:
    corpus = VECTOR_V1 / model / "full" / "corpus" / "vectors.npy"
    queries = (VECTOR_V2 if version == "v2" else VECTOR_V1) / model / "full" / "queries" / "vectors.npy"
    return corpus, queries


def validate_vector_metadata(model: str, kind: str, file: Path, texts: list[str]) -> None:
    """Catch old/misaligned caches when the embedding script stored fingerprints."""
    metadata_file = file.parent / "metadata.json"
    reuse_file = file.parent / "reuse_metadata.json"
    if metadata_file.is_file():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        settings = {
            "model": model,
            "model_id": MODEL_IDS[model],
            "kind": kind,
            "qwen_query_instruction": (
                "Instruct: Retrieve GIF captions that match the user's search query.\nQuery: "
                if model == "qwen3" else None
            ),
            "gemini_task": (
                "RETRIEVAL_QUERY" if kind == "queries" else "RETRIEVAL_DOCUMENT"
            ) if model == "gemini" else None,
            "texts": texts,
        }
        expected = hashlib.sha256(json.dumps(
            settings, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if metadata.get("fingerprint") and metadata["fingerprint"] != expected:
            raise ValueError(f"{model}/{kind}: vector metadata fingerprint differs from dataset; "
                             "wrong/stale cache! Please verify embeddings.")
        if metadata.get("count") != len(texts) or metadata.get("model") != model:
            raise ValueError(f"{model}/{kind}: cached model/count does not match dataset")
    elif reuse_file.is_file():
        reuse = json.loads(reuse_file.read_text(encoding="utf-8"))
        if reuse.get("model") != model or reuse.get("kind") != kind or reuse.get("count") != len(texts):
            raise ValueError(f"{model}/{kind}: reused cache metadata does not match")


def as_unit_vectors(file: Path, rows: int, name: str) -> np.ndarray:
    # Validate file existence and shape BEFORE starting to rank queries.
    vectors = np.load(file, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != rows:
        raise ValueError(f"{name}: expected {rows} rows, got {vectors.shape}")
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name}: contains NaN/Inf")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError(f"{name}: contains zero vectors")
    return vectors / lengths


def metrics(details: list[dict]) -> dict:
    n = len(details)
    if not n:
        return {"queries": 0, "recall_at_1": None, "recall_at_5": None, "mrr": None}
    ranks = [x["rank"] for x in details]
    return {
        "queries": n,
        "recall_at_1": round(sum(r == 1 for r in ranks) / n, 4),
        "recall_at_5": round(sum(r is not None and r <= TOP_K for r in ranks) / n, 4),
        "mrr": round(sum((1.0 / r) if r else 0.0 for r in ranks) / n, 4),
    }


def calculate_rankings(model: str, version: str, corpus: list[dict], queries: list[dict]) -> dict | None:
    corpus_file, query_file = vector_paths(model, version)
    missing = [str(p) for p in (corpus_file, query_file) if not p.is_file()]
    if missing:
        print(f"  NOT READY: missing saved vectors: {'; '.join(missing)}", flush=True)
        return None
    validate_vector_metadata(model, "corpus", corpus_file, [d["caption"] for d in corpus])
    validate_vector_metadata(model, "queries", query_file, [q["query"] for q in queries])
    corpus_vectors = as_unit_vectors(corpus_file, len(corpus), f"{model}: corpus")
    query_vectors = as_unit_vectors(query_file, len(queries), f"{model}: queries")
    if corpus_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError(f"{model}: corpus/query vector dimensions do not match")

    gif_ids = [d["gif_id"] for d in corpus]
    unique_gif_count = len(set(gif_ids))
    details = []
    for i, q in enumerate(queries):
        scores = corpus_vectors @ query_vectors[i]  # cosine similarity
        correct_ids = set(q["relevant_ids"])
        seen_gifs: set[str] = set()
        top5 = []
        correct_rank = None
        # Rank UNIQUE GIFs; for a GIF with two captions, its best caption wins.
        for row_idx in np.argsort(-scores, kind="stable"):
            index = int(row_idx)
            gif_id = gif_ids[index]
            if gif_id in seen_gifs:
                continue
            seen_gifs.add(gif_id)
            rank = len(seen_gifs)
            if rank <= TOP_K:
                d = corpus[index]
                top5.append({
                    "rank": rank,
                    "gif_id": gif_id,
                    "doc_id": d["doc_id"],
                    "caption": d["caption"],
                    "url": d.get("url", ""),
                    "score": round(float(scores[index]), 6),
                    "correct": gif_id in correct_ids,
                })
            if gif_id in correct_ids:
                correct_rank = rank
                # We have top 5 and rank of correct GIF, so we can stop.
                if len(top5) >= min(TOP_K, unique_gif_count):
                    break
            if correct_rank is not None and len(top5) >= TOP_K:
                break
        details.append({
            "query_id": q["query_id"],
            "source": get_source(q),
            "query": q["query"],
            "relevant_ids": q["relevant_ids"],
            "rank": correct_rank,
            "top5": top5,
        })
        if (i + 1) % 50 == 0 or i + 1 == len(queries):
            print(f"  Ranked {i + 1}/{len(queries)} queries", flush=True)

    report = {
        "model": MODEL_IDS[model],
        "model_key": model,
        "dataset": version,
        "corpus_sha256": file_digest(DATASETS[version] / "corpus.jsonl"),
        "queries_sha256": file_digest(DATASETS[version] / "queries.jsonl"),
        "corpus_captions": len(corpus),
        "unique_gifs": len(set(gif_ids)),
        "overall": metrics(details),
        "source_a": metrics([d for d in details if d["source"] == "source_a"]),
        "source_b": metrics([d for d in details if d["source"] == "source_b"]),
        "details": details,
    }
    return report


def ms_stats(values: list[float]) -> dict:
    return {
        "mean_ms": round(float(np.mean(values)), 3),
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
    }


def measure_latency(model: str, version: str, corpus: list[dict], queries: list[dict], count: int,
                    allow_gemini_api: bool = False, gemini_delay: float = 6.0) -> dict:
    """Explicit opt-in: re-encode queries one-by-one, reuse cached CORPUS vectors."""
    if model == "gemini" and not allow_gemini_api:
        raise ValueError("Gemini timing calls API; add --allow-gemini-api if you accept quota/possible cost")
    corpus_file, _ = vector_paths(model, version)
    corpus_vectors = as_unit_vectors(corpus_file, len(corpus), f"{model} corpus")
    gif_ids = [d["gif_id"] for d in corpus]
    unique_gif_count = len(set(gif_ids))
    selected = np.random.default_rng(42).choice(len(queries), size=min(count, len(queries)), replace=False)

    if model == "gemini":
        from google import genai
        from google.genai import types
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY in the terminal before measuring Gemini latency")
        client = genai.Client(api_key=key)
        last_request = 0.0

        def encoder(text: str) -> np.ndarray:
            nonlocal last_request
            response = client.models.embed_content(
                model=MODEL_IDS["gemini"], contents=[text],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            last_request = time.perf_counter()
            return np.asarray(response.embeddings[0].values, dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading {MODEL_NAMES[model]} onto CPU (excluded from timing)...", flush=True)
        st = SentenceTransformer(MODEL_IDS[model], device="cpu", trust_remote_code=False)

        def encoder(text: str) -> np.ndarray:
            if model == "qwen3":
                text = "Instruct: Retrieve GIF captions that match the user's search query.\nQuery: " + text
            options = dict(batch_size=1, convert_to_numpy=True,
                           normalize_embeddings=False, show_progress_bar=False)
            if model == "qwen3":
                options["prompt"] = ""
            return np.asarray(st.encode([text], **options)[0], dtype=np.float32)

    def search(query: str) -> tuple[float, float, float]:
        # When Gemini is selected, the optional quota-friendly delay occurs
        # before measuring the request (so network+inference, not artificial sleep).
        if model == "gemini" and last_request:
            elapsed = time.perf_counter() - last_request
            if elapsed < gemini_delay:
                time.sleep(gemini_delay - elapsed)
        start = time.perf_counter()
        v = encoder(query)
        after_embed = time.perf_counter()
        vnorm = np.linalg.norm(v)
        if not np.isfinite(vnorm) or vnorm == 0 or v.shape[0] != corpus_vectors.shape[1]:
            raise ValueError(f"{model}: invalid real-time query vector")
        scores = corpus_vectors @ (v / vnorm)
        seen = set()
        for row_idx in np.argsort(-scores, kind="stable"):
            seen.add(gif_ids[int(row_idx)])
            if len(seen) >= min(TOP_K, unique_gif_count):
                break
        end = time.perf_counter()
        return ((after_embed - start) * 1000,
                (end - after_embed) * 1000,
                (end - start) * 1000)

    print("  Warm-up, not counted...", flush=True)
    search(queries[int(selected[0])]["query"])
    observations = []
    for count_so_far, idx in enumerate(selected, 1):
        q = queries[int(idx)]
        embedding_ms, search_ms, total_ms = search(q["query"])
        observations.append({
            "query_id": q["query_id"],
            "embedding_ms": round(embedding_ms, 3),
            "search_ms": round(search_ms, 3),
            "total_ms": round(total_ms, 3),
        })
        if count_so_far % 20 == 0 or count_so_far == len(selected):
            print(f"  Timed {count_so_far}/{len(selected)}", flush=True)
    return {
        "model": model,
        "dataset": version,
        "device": "API" if model == "gemini" else "cpu",
        "system": platform.platform(),
        "queries_measured": len(observations),
        "warmup_queries": 1,
        "sampling": "random_without_replacement_seed_42",
        "search_top_k": TOP_K,
        "unit": "milliseconds_per_query",
        "excludes": ["model_loading", "corpus_vector_loading", "warmup", "intentional_API_delay"],
        "embedding": ms_stats([d["embedding_ms"] for d in observations]),
        "search": ms_stats([d["search_ms"] for d in observations]),
        "total": ms_stats([d["total_ms"] for d in observations]),
        "details": observations,
    }


def read_existing_latency(version: str, model: str, queries: list[dict]) -> dict | None:
    path = ROOT / "benchmark" / "results" / "dataset1" / "semantic" / f"{model}_latency.json"
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("queries_measured", 0) < 1 or not all(
                name in report and "mean_ms" in report[name]
                for name in ("embedding", "search", "total")):
            return None
        # Saved latency came from this dataset's folder; validate query IDs too.
        ids = {q["query_id"] for q in queries}
        if any(d["query_id"] not in ids for d in report.get("details", [])):
            return None
        return report
    except (ValueError, KeyError, TypeError, OSError):
        return None


def summarize_latencies(lat: dict | None) -> dict | None:
    if lat is None:
        return None
    return {
        "queries_measured": lat["queries_measured"],
        "embedding": lat["embedding"],
        "search": lat["search"],
        "total": lat["total"],
        "details_file": "",  # populated when a matching latency file exists
    }


def source_summary(report: dict) -> str:
    overall = report["overall"]
    return (f"R@1={overall['recall_at_1']:.1%} | "
            f"R@5={overall['recall_at_5']:.1%} | MRR={overall['mrr']:.4f}")


def answer_index(corpus: list[dict]) -> dict:
    grouped = defaultdict(list)
    for d in corpus:
        grouped[d["gif_id"]].append(d)
    return grouped


def merged_rows(queries: list[dict], corpus: list[dict], reports: dict) -> list[dict]:
    by_gif = answer_index(corpus)
    report_by_id = {name: {d["query_id"]: d for d in report["details"]}
                    for name, report in reports.items()}
    rows = []
    for q in queries:
        answers = []
        for gif_id in q["relevant_ids"]:
            captions = by_gif.get(gif_id, [])
            answers.append({
                "gif_id": gif_id,
                "captions": [d["caption"] for d in captions],
                "url": next((d.get("url") for d in captions if d.get("url")), ""),
            })
        rows.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "source": get_source(q),
            "split": q.get("split", ""),
            "answers": answers,
            "models": {name: report_by_id[name].get(q["query_id"]) if name in report_by_id else None
                       for name in ALL_MODELS},
        })
    return rows


def export_wide_csv(path: Path, rows: list[dict]) -> None:
    base_fields = ["query_id", "query_description", "source", "split",
                   "answer_gif_id", "answer_caption_in_corpus", "answer_gif_url"]
    fields = list(base_fields)
    for model in ALL_MODELS:
        fields.extend([f"{model}_status", f"{model}_correct_rank",
                       f"{model}_top1_correct", f"{model}_top5_correct"])
        for rank in range(1, TOP_K + 1):
            fields.extend([f"{model}_top{rank}_gif_id", f"{model}_top{rank}_caption",
                           f"{model}_top{rank}_score", f"{model}_top{rank}_correct",
                           f"{model}_top{rank}_gif_url"])
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = {
                "query_id": row["query_id"],
                "query_description": row["query"],
                "source": row["source"], "split": row["split"],
                "answer_gif_id": "; ".join(str(a["gif_id"]) for a in row["answers"]),
                "answer_caption_in_corpus": " | ".join(
                    " / ".join(str(c) for c in a["captions"]) for a in row["answers"]),
                "answer_gif_url": "; ".join(str(a["url"]) for a in row["answers"]),
            }
            for model in ALL_MODELS:
                item = row["models"].get(model)
                data[f"{model}_status"] = "ready" if item else "not_ready"
                if item:
                    rank = item["rank"]
                    data[f"{model}_correct_rank"] = rank if rank is not None else ""
                    data[f"{model}_top1_correct"] = int(rank == 1)
                    data[f"{model}_top5_correct"] = int(rank is not None and rank <= TOP_K)
                    for top in item["top5"]:
                        prefix = f"{model}_top{top['rank']}_"
                        data[prefix + "gif_id"] = top["gif_id"]
                        data[prefix + "caption"] = top["caption"]
                        data[prefix + "score"] = top["score"]
                        data[prefix + "correct"] = int(top["correct"])
                        data[prefix + "gif_url"] = top.get("url", "")
            writer.writerow(data)


def export_long_csv(path: Path, rows: list[dict]) -> None:
    fields = ["query_id", "query_description", "answer_gif_id", "answer_caption_in_corpus",
              "model", "correct_gif_rank", "top1_correct", "top5_correct",
              "retrieved_rank", "retrieved_gif_id", "retrieved_caption", "cosine_similarity",
              "correct_gif", "retrieved_gif_url"]
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for q in rows:
            for model in ALL_MODELS:
                item = q["models"].get(model)
                if not item:
                    continue
                for top in item["top5"]:
                    writer.writerow({
                        "query_id": q["query_id"],
                        "query_description": q["query"],
                        "answer_gif_id": "; ".join(str(a["gif_id"]) for a in q["answers"]),
                        "answer_caption_in_corpus": " | ".join(
                            " / ".join(str(c) for c in a["captions"]) for a in q["answers"]),
                        "model": model,
                        "correct_gif_rank": item["rank"] if item["rank"] is not None else "",
                        "top1_correct": int(item["rank"] == 1),
                        "top5_correct": int(item["rank"] is not None and item["rank"] <= TOP_K),
                        "retrieved_rank": top["rank"],
                        "retrieved_gif_id": top["gif_id"],
                        "retrieved_caption": top["caption"],
                        "cosine_similarity": top["score"],
                        "correct_gif": int(top["correct"]),
                        "retrieved_gif_url": top.get("url", ""),
                    })


def export_summary_csv(path: Path, reports: dict, latencies: dict) -> None:
    fields = ["model", "status", "queries", "recall_at_1_pct", "recall_at_5_pct", "mrr",
              "latency_queries", "embedding_mean_ms", "search_mean_ms", "total_mean_ms",
              "total_p50_ms", "total_p95_ms"]
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for model in ALL_MODELS:
            report, latency = reports.get(model), latencies.get(model)
            data = {"model": model, "status": "ready" if report else "not_ready"}
            if report:
                m = report["overall"]
                data.update(queries=m["queries"], recall_at_1_pct=round(100 * m["recall_at_1"], 2),
                            recall_at_5_pct=round(100 * m["recall_at_5"], 2), mrr=m["mrr"])
            if latency and report:
                data.update(latency_queries=latency["queries_measured"],
                            embedding_mean_ms=latency["embedding"]["mean_ms"],
                            search_mean_ms=latency["search"]["mean_ms"],
                            total_mean_ms=latency["total"]["mean_ms"],
                            total_p50_ms=latency["total"]["p50_ms"],
                            total_p95_ms=latency["total"]["p95_ms"])
            writer.writerow(data)


def safe_link(url: object) -> str:
    value = str(url or "")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return value


def gif_card(gif_id: object, caption: object, url: object, *, rank: int | None = None,
             similarity: float | None = None, correct: bool | None = None) -> str:
    title = "GIF đáp án" if rank is None else f"Top {rank}"
    state = "" if correct is None else ('<span class="correct">ĐÚNG GIF</span>' if correct
                                         else '<span class="incorrect">Khác GIF ID</span>')
    score = f'<span class="score">Cosine {similarity:.4f}</span>' if similarity is not None else ""
    link = safe_link(url)
    if link:
        attr = html.escape(link, quote=True)
        img = f'<a href="{attr}" target="_blank" rel="noopener noreferrer"><img src="{attr}" loading="lazy" alt="GIF"/></a>'
    else:
        img = '<div class="no-image">Không có link GIF</div>'
    return (f'<article class="gif-card"><div class="gif-head"><strong>{title}</strong>{state}{score}</div>'
            f'{img}<div class="gif-id">ID: {html.escape(str(gif_id))}</div>'
            f'<div class="gif-caption">{html.escape(str(caption))}</div></article>')


def export_html(path: Path, rows: list[dict], reports: dict, latencies: dict) -> None:
    summary_head = []
    for model in ALL_MODELS:
        report, latency = reports.get(model), latencies.get(model)
        if report:
            m = report["overall"]
            time_label = f"{latency['total']['mean_ms']:.1f} ms" if latency else "—"
            summary_head.append(
                f'<tr><th>{html.escape(MODEL_NAMES[model])}</th>'
                f'<td>{m["recall_at_1"]:.1%}</td><td>{m["recall_at_5"]:.1%}</td>'
                f'<td>{m["mrr"]:.4f}</td><td>{time_label}</td><td>{latency["queries_measured"] if latency else "—"}</td></tr>'
            )
        else:
            summary_head.append(f'<tr><th>{html.escape(MODEL_NAMES[model])}</th><td colspan="5">Chưa có đủ embeddings</td></tr>')

    query_blocks = []
    for q in rows:
        model_items = []
        all_wrong = True
        any_wrong = False
        for model in ALL_MODELS:
            item = q["models"].get(model)
            if not item:
                model_items.append(f'<section class="model-panel missing"><h3>{MODEL_NAMES[model]}</h3><p>Chưa sẵn sàng (thiếu vectors).</p></section>')
                continue
            right_at_one = item["rank"] == 1
            all_wrong = all_wrong and not right_at_one
            any_wrong = any_wrong or not right_at_one
            rank_label = f'Thứ hạng GIF đúng: {item["rank"]}' if item["rank"] is not None else 'GIF đúng chưa có hạng'
            badge = '<span class="pill good">Top 1 đúng</span>' if right_at_one else '<span class="pill bad">Top 1 sai</span>'
            cards = ''.join(gif_card(t["gif_id"], t["caption"], t.get("url"), rank=t["rank"],
                                     similarity=t["score"], correct=t["correct"])
                            for t in item["top5"])
            model_items.append(f'<section class="model-panel"><header><h3>{MODEL_NAMES[model]}</h3>{badge}</header>'
                               f'<div class="rank-label">{html.escape(rank_label)}</div>{cards}</section>')
        # Avoid classifying unready Gemini as an incorrect model.
        available = [name for name in ALL_MODELS if q["models"].get(name) is not None]
        all_wrong = bool(available) and all(q["models"][name]["rank"] != 1 for name in available)
        any_wrong = any(q["models"][name]["rank"] != 1 for name in available)
        answer_html = ''.join(gif_card(ans["gif_id"], " / ".join(ans["captions"]), ans["url"])
                              for ans in q["answers"])
        status = 'all_wrong' if all_wrong else ('some_wrong' if any_wrong else 'all_correct')
        preview = html.escape(q["query"])
        query_blocks.append(
            f'<details class="query" data-status="{status}" data-search="{html.escape(str(q["query_id"]) + " " + q["query"], quote=True)}">'
            f'<summary><span class="qid">{html.escape(str(q["query_id"]))}</span>'
            f'<span class="qtext">{preview}</span>'
            f'<span class="pill {"bad" if all_wrong else "neutral"}">{"Tất cả sai Top 1" if all_wrong else "Xem kết quả"}</span></summary>'
            f'<div class="query-body"><h3>Đáp án theo dataset</h3><div class="answers">{answer_html}</div>'
            f'<h3>Top 5 của từng model</h3><div class="model-grid">{"".join(model_items)}</div></div></details>'
        )

    html_doc = '''<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GIF Retrieval — Dataset 1 V2 — Error Analysis</title>
<style>
:root{color-scheme:light;font-family:system-ui,Segoe UI,Arial,sans-serif;background:#f4f6f9;color:#1e293b}
*{box-sizing:border-box}body{max-width:1460px;margin:0 auto;padding:20px}
h1{font-size:27px;margin-bottom:6px}h2{font-size:19px}h3{font-size:15px;margin:12px 0}
.subtle{color:#64748b;font-size:14px}.panel{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:18px;margin:16px 0}
table{border-collapse:collapse;width:100%;min-width:550px}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #e2e8f0}
.table-scroll{overflow-x:auto}.tools{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.tools input,.tools select{padding:10px;border-radius:8px;border:1px solid #cbd5e1;font:inherit}
.tools input{min-width:280px;flex:1}.tools button{padding:10px;border-radius:8px;border:1px solid #cbd5e1;background:#fff;cursor:pointer}
.query{border:1px solid #e2e8f0;background:#fff;border-radius:13px;margin:12px 0;overflow:hidden}
.query summary{cursor:pointer;padding:14px;display:flex;flex-wrap:wrap;gap:14px;align-items:center}
.qid{font-size:12px;font-weight:800;color:#475569}.qtext{font-weight:600;flex:1;min-width:240px}
.query-body{border-top:1px solid #e2e8f0;padding:16px}.pill{font-size:11px;font-weight:700;padding:4px 9px;border-radius:15px;white-space:nowrap}
.good,.correct{background:#dcfce7;color:#166534}.bad,.incorrect{background:#fee2e2;color:#991b1b}.neutral{background:#f1f5f9;color:#475569}
.answers{display:flex;max-width:480px;flex-wrap:wrap;gap:10px}.answers .gif-card{width:100%}
.model-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}
.model-panel{border:1px solid #dbe3ed;border-radius:10px;padding:10px;min-width:0;background:#fbfcff}
.model-panel header{display:flex;gap:9px;align-items:center;justify-content:space-between}
.model-panel h3{margin:2px 0}.model-panel.missing{color:#64748b}.rank-label{font-size:12px;color:#64748b;margin-bottom:10px}
.gif-card{border:1px solid #e2e8f0;border-radius:9px;padding:9px;margin:8px 0;background:#fff;min-width:0}
.gif-head{display:flex;flex-wrap:wrap;gap:7px;align-items:center;font-size:12px;margin-bottom:5px}
.correct,.incorrect{border-radius:12px;padding:2px 6px;font-size:10px;font-weight:700}.score{color:#64748b;font-size:11px}
.gif-card img{width:100%;max-height:145px;object-fit:contain;background:#f1f5f9;border-radius:6px}
.no-image{background:#f1f5f9;color:#64748b;padding:18px;text-align:center;border-radius:6px}
.gif-id{font-size:11px;font-weight:650;color:#475569;margin:6px 0;overflow-wrap:anywhere}.gif-caption{font-size:12px;line-height:1.45;overflow-wrap:anywhere}
[hidden]{display:none!important}#count{font-size:13px;color:#475569}
</style></head><body>
<h1>GIF Retrieval — So sánh kết quả từng query</h1>
<p class="subtle">Mỗi query có GIF đáp án từ Dataset và Top 5 của từng model. Đúng/sai được xét theo GIF ID gắn nhãn; GIF khác vẫn có thể phù hợp về nghĩa.</p>
<section class="panel"><h2>Tổng hợp Recall, MRR và latency</h2><div class="table-scroll"><table><thead><tr><th>Model</th><th>R@1</th><th>R@5</th><th>MRR</th><th>Latency mean</th><th>Số query đo tốc độ</th></tr></thead><tbody>
__SUMMARY__
</tbody></table></div><p class="subtle">Accuracy dùng vector có sẵn trên toàn bộ tập query. Latency lấy từ kết quả đã đo trước đó (nếu có), không gọi lại model.</p></section>
<section class="panel"><h2>Chi tiết từng query (200 queries trong Dataset V2)</h2><div class="tools">
<input id="search" type="search" placeholder="Tìm query ID hoặc mô tả..." aria-label="Tìm query"/>
<select id="status"><option value="all">Tất cả queries</option><option value="all_wrong">Tất cả model có kết quả đều sai Top 1</option><option value="some_wrong">Ít nhất một model sai Top 1</option><option value="all_correct">Tất cả model có kết quả đúng Top 1</option></select>
<button id="expand" type="button">Mở tất cả (đang lọc)</button><button id="collapse" type="button">Thu gọn</button><span id="count"></span>
</div></section>
__QUERIES__
<script>
const items=Array.from(document.querySelectorAll('details.query'));
const search=document.getElementById('search'), status=document.getElementById('status'), counter=document.getElementById('count');
function update(){const text=search.value.toLocaleLowerCase().trim(), value=status.value;
  let shown=0;for(const item of items){const searchable=item.getAttribute('data-search').toLocaleLowerCase();
    const validStatus=value==='all'||(value==='some_wrong' ? item.dataset.status!=='all_correct' : item.dataset.status===value);
    item.hidden=!(validStatus&&searchable.includes(text));if(!item.hidden)shown++;}
  counter.textContent=shown+' / '+items.length+' queries';}
search.addEventListener('input',update);status.addEventListener('change',update);
document.getElementById('expand').addEventListener('click',()=>items.forEach(i=>{if(!i.hidden)i.open=true}));
document.getElementById('collapse').addEventListener('click',()=>items.forEach(i=>i.open=false));update();
</script></body></html>'''
    path.write_text(html_doc.replace("__SUMMARY__", "\n".join(summary_head)).replace(
        "__QUERIES__", "\n".join(query_blocks)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cached GIF embeddings; export CSV + HTML query-by-query comparison")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="v2", help="Default: v2 (corrected 200 Source A)")
    parser.add_argument("--model", choices=("all", *ALL_MODELS), default="all", help="Default: all; Gemini is skipped until vectors are ready")
    parser.add_argument("--measure-latency", action="store_true", help="Explicitly re-encode queries to measure fresh latency; slower")
    parser.add_argument("--latency-queries", type=int, default=200, help="Number of queries for fresh timing (default 200)")
    parser.add_argument("--allow-gemini-api", action="store_true", help="Allow extra Gemini API requests for timing (only with --measure-latency)")
    parser.add_argument("--gemini-delay", type=float, default=6.0, help="Seconds between optional Gemini latency API calls")
    parser.add_argument("--no-latency", action="store_true", help="Do not display saved latency (compatibility)")
    parser.add_argument("--latency", action="store_true", help="Legacy flag: same as --measure-latency")
    parser.add_argument("--inspect", type=int, default=0, help="Legacy flag: report always includes all queries")
    parser.add_argument("--errors-only", action="store_true", help="Use HTML filter 'Tất cả model sai Top 1' after opening")
    parser.add_argument("--query-id", type=str, help="To find one query, use HTML search after opening")
    args = parser.parse_args()
    if args.latency_queries < 1 or args.gemini_delay < 0 or args.inspect < 0:
        parser.error("--latency-queries must be >=1; --gemini-delay and --inspect must be >=0")
    if args.latency:
        args.measure_latency = True
    if args.allow_gemini_api and not args.measure_latency:
        parser.error("--allow-gemini-api requires --measure-latency")

    corpus, queries = load_dataset(args.dataset)
    folder = ROOT / "benchmark" / "results" / "dataset1" / "semantic"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Dataset {args.dataset}: {len(corpus)} captions, "
          f"{len(set(d['gif_id'] for d in corpus))} GIFs, {len(queries)} queries.", flush=True)
    print("Default: accuracy from saved vectors; saved latency reused; NO API calls.", flush=True)
    selected_models = list(ALL_MODELS) if args.model == "all" else [args.model]
    reports = {}
    latencies = {}
    comparison = {}

    for model in selected_models:
        print(f"\n[{MODEL_NAMES[model]}]", flush=True)
        try:
            report = calculate_rankings(model, args.dataset, corpus, queries)
        except (OSError, ValueError, MemoryError) as exc:
            print(f"  ERROR: {exc}", flush=True)
            comparison[model] = {"status": "error", "reason": str(exc)}
            continue
        if report is None:
            comparison[model] = {"status": "not_ready"}
            continue
        reports[model] = report
        if not args.no_latency:
            latency = read_existing_latency(args.dataset, model, queries)
            if args.measure_latency:
                if model == "gemini" and not args.allow_gemini_api:
                    print("  Gemini latency not called (quota-protected); use --allow-gemini-api explicitly.")
                else:
                    try:
                        latency = measure_latency(model, args.dataset, corpus, queries,
                                                  args.latency_queries, args.allow_gemini_api,
                                                  args.gemini_delay)
                        write_json(folder / f"{model}_latency.json", latency)
                    except Exception as exc:
                        print(f"  LATENCY ERROR: {exc}; using existing latency if available")
            if latency:
                latencies[model] = latency
                report["latency"] = summarize_latencies(latency)
                report["latency"]["details_file"] = f"{model}_latency.json"
        write_json(folder / f"{model}_results.json", report)
        comparison[model] = {
            "status": "evaluated", "model": MODEL_IDS[model],
            "overall": report["overall"], "source_a": report["source_a"],
            "source_b": report["source_b"],
        }
        if model in latencies:
            comparison[model]["latency"] = summarize_latencies(latencies[model])
            l = latencies[model]
            print(f"  Latency mean={l['total']['mean_ms']:.1f} ms; "
                  f"p95={l['total']['p95_ms']:.1f} ms; "
                  f"{l['queries_measured']} queries (saved or measured)", flush=True)
        print("  " + source_summary(report), flush=True)

    if not reports:
        raise SystemExit("No model has complete, matching embeddings yet; no comparison was written.")
    write_json(folder / "comparison.json", comparison)
    rows = merged_rows(queries, corpus, reports)
    wide = folder / "query_comparison.csv"
    long = folder / "query_comparison_long.csv"
    summary = folder / "benchmark_summary.csv"
    gallery = folder / "query_comparison.html"
    export_wide_csv(wide, rows)
    export_long_csv(long, rows)
    export_summary_csv(summary, reports, latencies)
    export_html(gallery, rows, reports, latencies)

    print("\nEXPORTS (no embeddings changed):", flush=True)
    print(f"  {summary}         (one row per model)")
    print(f"  {wide}         (one row per query; ALL models side by side)")
    print(f"  {long}    (one row per Top-5 result; easy filters)")
    print(f"  {gallery}        (open in browser to view GIFs)")
    print(f"  Models ready: {len(reports)}/{len(selected_models)}; missing Gemini will show N/A until completed.")


if __name__ == "__main__":
    main()
