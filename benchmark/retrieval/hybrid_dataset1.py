"""Run a single weighted BM25 + semantic retrieval experiment on Dataset 1.

From GIFt-main root:
    python -m benchmark.retrieval.hybrid_dataset1

To change the experiment, edit only the three DEFAULT_* constants below, save,
and run the SAME command again. Keyword weight = 1 - semantic weight.

All 200 queries are evaluated against 10,000 captions, ranked at unique GIF level.
Every run is saved separately; the independent comparison script is called after
saving to update CSV/HTML for all historical runs.

Reads saved embeddings only. Does not load models, download files or call APIs.
Hybrid timing measures retrieval + fusion, not query encoding or index building.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import time

import numpy as np

from benchmark.retrieval.keyword_dataset1 import BM25
from benchmark.retrieval.semantic_dataset1 import (
    as_unit_vectors, get_source, load_dataset, metrics,
    validate_vector_metadata, vector_paths,
)

# ====== EDIT THESE THREE VALUES TO SET YOUR DEFAULT EXPERIMENT ======
DEFAULT_SEMANTIC_MODEL = "qwen3"    # "minilm", "qwen3", or "bge-m3"
DEFAULT_SEMANTIC_WEIGHT = 0.80     # 0.70 semantic + automatically 0.30 BM25
DEFAULT_RRF_K = 60                 # typical RRF rank offset
# ===================================================================

ALLOWED_MODELS = ("minilm", "qwen3", "bge-m3")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmark" / "datasets" / "dataset1"
OUT = ROOT / "benchmark" / "results" / "dataset1" / "hybrid"
RUNS = OUT  # Each experiment is stored directly in hybrid/<run_id>/
TOP_K = 5
PROTOCOL = "gif-unique-weighted-rrf-v1"


def validate_weight(weight: float | str) -> float:
    try:
        number = Decimal(str(weight))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("semantic_weight must be a number in [0, 1]") from exc
    if not number.is_finite() or not (Decimal("0") <= number <= Decimal("1")):
        raise ValueError("semantic_weight must be between 0 and 1 (inclusive)")
    return float(number)


def gif_ranks(scores, corpus):
    """GIF -> (one-based rank, best caption row), stable under equal scores."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or len(scores) != len(corpus) or not np.isfinite(scores).all():
        raise ValueError("Score array must match corpus length and contain finite values")
    result = {}
    for i in np.argsort(-scores, kind="stable"):
        idx = int(i)
        gid = corpus[idx]["gif_id"]
        if gid not in result:
            result[gid] = (len(result) + 1, idx)
    return result


def weighted_fusion(query, corpus, bm25, semantic_scores, correct_ids,
                    semantic_weight=DEFAULT_SEMANTIC_WEIGHT, k=DEFAULT_RRF_K):
    """Full-corpus Weighted RRF by unique GIF rank, not by raw similarity.

    score(g) = semantic_weight / (k + semantic_rank(g))
             + (1 - semantic_weight) / (k + bm25_rank(g)).

    k=60 and equal weights have exactly the ordering of classic unweighted RRF.
    """
    alpha = validate_weight(semantic_weight)
    if not isinstance(k, int) or k <= 0:
        raise ValueError("rrf_k must be a positive integer")
    lex_scores = bm25.scores(query)
    lex = gif_ranks(lex_scores, corpus)
    sem = gif_ranks(semantic_scores, corpus)
    if lex.keys() != sem.keys():
        raise ValueError("BM25 and semantic rankings do not contain identical GIF IDs")
    fusion = {
        gid: alpha / (k + sem[gid][0]) + (1.0 - alpha) / (k + lex[gid][0])
        for gid in sem
    }
    # Full ranking, even when the labeled GIF is not in Top 5 (correct MRR).
    # Deterministic tie-breaks for reproducibility.
    ordered = sorted(fusion, key=lambda gid: (-fusion[gid], sem[gid][0], lex[gid][0], gid))
    target = set(correct_ids)
    correct_rank = next((i for i, gid in enumerate(ordered, 1) if gid in target), None)
    top = []
    for rank, gid in enumerate(ordered[:TOP_K], 1):
        # Show the best caption from the stronger branch.
        doc_idx = sem[gid][1] if alpha >= 0.5 else lex[gid][1]
        doc = corpus[doc_idx]
        top.append({
            "rank": rank, "gif_id": gid, "doc_id": doc["doc_id"],
            "caption": doc["caption"], "url": doc.get("url", ""),
            "score": round(float(fusion[gid]), 10), "correct": gid in target,
            "semantic_rank": sem[gid][0], "bm25_rank": lex[gid][0],
        })
    return top, correct_rank


def hybrid_one(query, corpus, bm25, semantic_scores, correct_ids, k=60,
               semantic_weight=0.5):
    """Compatibility helper for old tests; also accepts semantic_weight."""
    return weighted_fusion(query, corpus, bm25, semantic_scores, correct_ids,
                           semantic_weight=semantic_weight, k=k)


def atomic_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(model=DEFAULT_SEMANTIC_MODEL, semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
        k=DEFAULT_RRF_K):
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Model {model!r} is unsupported; choose from {ALLOWED_MODELS}")
    alpha = validate_weight(semantic_weight)
    if not isinstance(k, int) or k <= 0:
        raise ValueError("rrf_k must be a positive integer")
    beta = round(1.0 - alpha, 12)
    corpus, queries = load_dataset("v2")
    if len(corpus) != 10000 or len(queries) != 200:
        raise ValueError("Expected exactly 10,000 captions and 200 queries")
    cp, qp = vector_paths(model, "v2")
    for path in (cp, qp):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing cached embeddings: {path}\n"
                f"Run from project root: python -m benchmark embed --model {model} --full"
            )
    validate_vector_metadata(model, "corpus", cp, [d["caption"] for d in corpus])
    validate_vector_metadata(model, "queries", qp, [q["query"] for q in queries])
    cv = as_unit_vectors(cp, len(corpus), f"{model} corpus")
    qv = as_unit_vectors(qp, len(queries), f"{model} queries")
    if cv.shape[1] != qv.shape[1]:
        raise ValueError("Cached corpus and query vectors have different dimensions")

    print(f"Running BM25 + {model}: semantic={alpha:.2%}, BM25={beta:.2%}, RRF k={k}", flush=True)
    bm25 = BM25(corpus)
    details, times = [], []
    for i, q in enumerate(queries):
        start = time.perf_counter()
        semantic_scores = cv @ qv[i]
        top, rank = weighted_fusion(q["query"], corpus, bm25, semantic_scores,
                                    q["relevant_ids"], alpha, k)
        ms = (time.perf_counter() - start) * 1000.0
        times.append({"query_id": q["query_id"], "retrieval_fusion_ms": round(ms, 4)})
        details.append({
            "query_id": q["query_id"], "query": q["query"],
            "source": get_source(q), "relevant_ids": q["relevant_ids"],
            "rank": rank, "top5": top,
        })
        if (i + 1) % 50 == 0:
            print(f"Ranked {i + 1}/{len(queries)} queries", flush=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    weight_tag = f"{alpha:.4f}".replace(".", "p")
    run_id = f"{timestamp}_{model}_sw{weight_tag}_k{k}_{secrets.token_hex(3)}"
    latency = {
        "type": "offline_retrieval_and_fusion_only",
        "mean_ms": round(sum(t["retrieval_fusion_ms"] for t in times) / len(times), 4),
        "queries_measured": len(times),
        "excludes": ["query_embedding", "model_loading", "corpus_embedding", "BM25_index_building"],
        "note": "NOT directly comparable to semantic query-embedding + search latency",
    }
    report = {
        "protocol": PROTOCOL,
        "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": f"BM25 + {model} (semantic={alpha:g}, BM25={beta:g}, RRF k={k})",
        "model_key": "hybrid", "semantic_model": model,
        "semantic_weight": alpha, "keyword_weight": beta, "rrf_k": k,
        "bm25_parameters": {"k1": bm25.k1, "b": bm25.b},
        "dataset": "v2", "corpus_captions": len(corpus),
        "unique_gifs": bm25.unique_gif_count,
        "corpus_sha256": hashlib.sha256((DATA / "corpus.jsonl").read_bytes()).hexdigest(),
        "queries_sha256": hashlib.sha256((DATA / "queries.jsonl").read_bytes()).hexdigest(),
        "ranking_unit": "unique_gif_id", "fusion": "weighted_rrf_full_ranking",
        "overall": metrics(details),
        "source_a": metrics([d for d in details if d["source"] == "source_a"]),
        "source_b": metrics([d for d in details if d["source"] == "source_b"]),
        "latency": latency, "details": details,
    }

    RUNS.mkdir(parents=True, exist_ok=True)
    final_dir = RUNS / run_id
    stage_dir = RUNS / (".incomplete_" + run_id)
    stage_dir.mkdir(exist_ok=False)
    try:
        atomic_json(stage_dir / "hybrid_results.json", report)
        atomic_json(stage_dir / "hybrid_latency.json", {
            "run_id": run_id, "model": model, **latency, "measurements": times,
        })
        # A complete experiment appears in runs/ only after all files are written.
        stage_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    print(f"Saved experiment: {final_dir}", flush=True)
    print(f"R@1 = {report['overall']['recall_at_1']:.1%}; "
          f"R@5 = {report['overall']['recall_at_5']:.1%}; "
          f"MRR = {report['overall']['mrr']:.4f}", flush=True)
    from benchmark.reports.compare_dataset1 import build_report
    try:
        build_report()  # Scan ALL historical experiments, refresh CSV/HTML.
    except Exception as exc:
        print(f"WARNING: Hybrid saved, but comparison update failed: {exc}", flush=True)
        print("Run python -m benchmark compare1 after fixing the error.", flush=True)
    return report


def main():
    """Run the configuration above with no command-line flags."""
    run(
        model=DEFAULT_SEMANTIC_MODEL,
        semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
        k=DEFAULT_RRF_K,
    )


if __name__ == "__main__":
    main()
