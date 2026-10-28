"""Dataset 2: BM25 + semantic Weighted RRF over 96 three-choice text cases.

Run at GIFt-main root, without CLI flags:
    python -m benchmark.retrieval.hybrid_dataset2

ONLY change the three DEFAULT_* constants below to choose model/weights/RRF-k.
The semantic scores come from saved Dataset 2 semantic/ranking_<model>.csv;
no new embedding models are run. The BM25 branch uses the same 3-choice
candidate indexing as keyword_dataset2.py.

Every run is saved separately under benchmark/results/dataset2/hybrid/<run_id>/
and automatically updates benchmark/results/dataset2/comparison/ CSV/HTML.
Dataset 1 output and Dataset 2 semantic output are never modified.

IMPORTANT: Dataset 2 measures accuracy ranking 1 positive vs 2 negatives;
this is NOT R@5 over the 10k GIF corpus.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import secrets
import shutil
import time

from benchmark.datasets.edge_cases import TEST_CASES
from benchmark.retrieval.keyword_dataset2 import (
    HYBRID_DIR, SEMANTIC_DIR, BASELINE_NAMES,
    BM25_K1, BM25_B, RESULT_COLUMNS, bm25_scores, build_comparison,
    category_metrics, checked_ranking, dataset_hash, make_row, metric,
    rank_indices, validate_cases, write_csv,
)

# ================= CHANGE ONLY THESE THREE VALUES =================
DEFAULT_SEMANTIC_MODEL = "qwen3"    # "minilm", "qwen3", "bge-m3"
DEFAULT_SEMANTIC_WEIGHT = 0.50      # 70% semantic; automatically 30% BM25
DEFAULT_RRF_K = 60                 # Reciprocal Rank Fusion offset
# ================================================================

ALLOWED_MODELS = ("minilm", "qwen3", "bge-m3")
PROTOCOL = "edge-3choice-weighted-rrf-v1"


def validate_weight(value) -> float:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("semantic weight must be a number from 0 to 1") from exc
    if not number.is_finite() or not Decimal("0") <= number <= Decimal("1"):
        raise ValueError("semantic weight must be in [0,1]")
    return float(number)


def fusion_scores(semantic_scores: list[float], keyword_scores: list[float],
                  semantic_weight: float, k: int) -> list[float]:
    """Weighted RRF on 3 candidate rankings, ties favor negatives.

    f_i = a/(k + semantic_rank_i) + (1-a)/(k + BM25_rank_i).
    Never add BM25 lexical raw scores to cosine similarity scores.
    """
    alpha = validate_weight(semantic_weight)
    if type(k) is not int or k <= 0:
        raise ValueError("RRF_K must be a positive integer")
    _, semantic_ranks = rank_indices(semantic_scores)
    _, keyword_ranks = rank_indices(keyword_scores)
    return [
        alpha / (k + semantic_ranks[i]) + (1.0 - alpha) / (k + keyword_ranks[i])
        for i in range(3)
    ]


def _load_semantic_rows(model: str, fingerprint: str) -> list[dict]:
    file = SEMANTIC_DIR / f"ranking_{model}.csv"
    rows = checked_ranking(file, model, fingerprint)
    if rows is None:
        raise FileNotFoundError(
            f"Missing or incompatible saved Dataset 2 semantic scores: {file}\n"
            f"Run: python -m benchmark.retrieval.semantic_dataset2 --model {model}\n"
            "No new embeddings are needed if the matching ranking CSV already exists."
        )
    return rows


def _atomic_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(model: str = DEFAULT_SEMANTIC_MODEL,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        k: int = DEFAULT_RRF_K) -> dict:
    validate_cases()
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported semantic model {model!r}; choose {ALLOWED_MODELS}")
    alpha = validate_weight(semantic_weight)
    beta = 1.0 - alpha
    if type(k) is not int or k <= 0:
        raise ValueError("RRF_K must be a positive integer")
    fingerprint = dataset_hash()
    semantic_rows = _load_semantic_rows(model, fingerprint)
    print(f"Dataset 2 Hybrid: BM25 + {model} | semantic={alpha:.1%}, BM25={beta:.1%}, RRF k={k}", flush=True)

    rows, timings = [], []
    for i, (case, semantic) in enumerate(zip(TEST_CASES, semantic_rows), 1):
        start = time.perf_counter()
        sem_scores = [float(semantic[c]) for c in
                      ("positive_score", "negative_1_score", "negative_2_score")]
        if not all(math.isfinite(x) for x in sem_scores):
            raise ValueError(f"Non-finite semantic scores for case {case['id']}")
        lex_scores = bm25_scores(case)
        fused = fusion_scores(sem_scores, lex_scores, alpha, k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        row = make_row(case, fused, "hybrid", fingerprint, elapsed_ms)
        # Retain the original branch ranks for detailed inspection.
        row["semantic_ranks"] = json.dumps(rank_indices(sem_scores)[1])
        row["bm25_ranks"] = json.dumps(rank_indices(lex_scores)[1])
        rows.append(row)
        timings.append({"id": case["id"], "latency_ms": round(elapsed_ms, 6)})
        if i % 24 == 0:
            print(f"Ranked {i}/{len(TEST_CASES)} cases", flush=True)

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    alpha_tag = f"{alpha:.4f}".replace(".", "p")
    run_id = f"{stamp}_{model}_sw{alpha_tag}_k{k}_{secrets.token_hex(3)}"
    latency = {
        "type": "BM25_three_choice_scoring_plus_RRF_only",
        "mean_ms": round(sum(t["latency_ms"] for t in timings) / len(timings), 6),
        "cases_measured": len(timings),
        "excludes": ["semantic_query_embedding", "semantic_candidate_embedding", "model_loading"],
        "note": "Do not compare directly to semantic batch embedding time.",
    }
    report = {
        "protocol": PROTOCOL, "dataset": "dataset2", "dataset_sha256": fingerprint,
        "run_id": run_id, "created_at_utc": now.isoformat(),
        "model": f"BM25 + {BASELINE_NAMES[model]} (semantic={alpha:g}, BM25={beta:g}, RRF k={k})",
        "model_key": "hybrid", "semantic_model": model,
        "semantic_weight": alpha, "keyword_weight": beta, "rrf_k": k,
        "bm25_parameters": {"k1": BM25_K1, "b": BM25_B},
        "ranking_unit": "one_positive_two_negatives_per_case",
        "fusion": "weighted_rrf_3_choice",
        "overall": metric(rows),
        "by_category": category_metrics(rows),
        "latency": latency,
    }

    # Publish a run only when every file has been written. Never overwrite history.
    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    final_dir = HYBRID_DIR / run_id
    stage_dir = HYBRID_DIR / (".incomplete_" + run_id)
    stage_dir.mkdir(exist_ok=False)
    try:
        write_csv(stage_dir / "ranking_hybrid.csv", rows,
                  (*RESULT_COLUMNS, "semantic_ranks", "bm25_ranks"))
        _atomic_json(stage_dir / "hybrid_results.json", report)
        _atomic_json(stage_dir / "hybrid_latency.json", {
            "run_id": run_id, "dataset_sha256": fingerprint, **latency,
            "measurements": timings,
        })
        stage_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    stats = report["overall"]
    print(f"SAVED: {final_dir}", flush=True)
    print(f"Hit@1: {stats['correct']}/96 ({stats['hit_at_1_pct']:.2f}%) | MRR: {stats['mrr']:.4f}", flush=True)
    try:
        build_comparison()
    except Exception as exc:
        print(f"WARNING: Hybrid results saved but comparison update failed: {exc}", flush=True)
        print("Rebuild: python -c \"from benchmark.retrieval.keyword_dataset2 import build_comparison; build_comparison()\"")
    return report


def main():
    run(DEFAULT_SEMANTIC_MODEL, DEFAULT_SEMANTIC_WEIGHT, DEFAULT_RRF_K)


if __name__ == "__main__":
    main()
