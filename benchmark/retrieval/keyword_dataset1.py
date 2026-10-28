"""BM25 keyword retrieval benchmark for GIFt Dataset 1 (10k captions, 200 queries).

Put this file at: GIFt-main/benchmark/keyworddata1.py
Run from GIFt-main: python -m benchmark.retrieval.keyword_dataset1

INPUTS (read-only):
    data/benchmark/dataset1/corpus.jsonl
    data/benchmark/dataset1/queries.jsonl

OUTPUTS (ALL in benchmark/results/dataset1/keyword/):
    bm25_results.json, bm25_latency.json, comparison.json,
    benchmark_summary.csv, query_comparison.csv,
    query_comparison_long.csv, query_comparison.html

Optional READ-ONLY comparison: if saved semantic {minilm,qwen3,bge-m3,gemini}_results.json
and matching *_latency.json exist in benchmark/results/dataset1/semantic/, include them in CSV/HTML.
NO embeddings, no ML models, no API calls, no third-party packages.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import re
import statistics
import time
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "benchmark"
DATASETS = {"v2": BASE / "dataset1"}
OUT = ROOT / "benchmark" / "results" / "dataset1" / "keyword"
MODEL_ORDER = ("bm25", "minilm", "qwen3", "bge-m3", "gemini")
MODEL_NAMES = {
    "bm25": "BM25 (Keyword)", "minilm": "MiniLM",
    "qwen3": "Qwen3", "bge-m3": "BGE-M3", "gemini": "Gemini",
}
TOP_K = 5
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Unicode word tokenizer; reproducible and independent of embedding models."""
    return TOKEN_PATTERN.findall(str(text).casefold())


def normalized_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def source_name(row: dict) -> str:
    raw = str(row.get("source", "")).lower().replace("-", "_").strip()
    return {"a": "source_a", "b": "source_b"}.get(raw, raw or "unknown")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {path}\n"
            "Dataset V2 phải có corpus.jsonl và queries.jsonl (tạo bằng build_dataset1.py trước)."
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_data(dataset: str) -> tuple[list[dict], list[dict]]:
    folder = DATASETS[dataset]
    corpus = read_jsonl(folder / "corpus.jsonl")
    queries = read_jsonl(folder / "queries.jsonl")
    if len(corpus) != 10_000 or len(queries) != 200:
        raise ValueError(
            f"Sai kích thước Dataset {dataset}: {len(corpus)} captions, {len(queries)} queries. "
            "Cần đúng 10.000 captions và 200 queries để so sánh công bằng."
        )
    doc_ids = [d["doc_id"] for d in corpus]
    doc_id_set = set(doc_ids)
    query_ids = [q["query_id"] for q in queries]
    if len(set(doc_ids)) != len(corpus) or len(set(query_ids)) != len(queries):
        raise ValueError("Trùng doc_id hoặc query_id trong dataset")
    if any(not isinstance(d.get("caption"), str) or not d.get("gif_id") for d in corpus):
        raise ValueError("Corpus có caption/gif_id không hợp lệ")
    gif_ids = {d["gif_id"] for d in corpus}
    for q in queries:
        ids = q.get("relevant_ids")
        if not isinstance(ids, list) or not ids or not set(ids).issubset(gif_ids):
            raise ValueError(f"Ground truth GIF không nằm trong corpus: {q['query_id']}")
        if not isinstance(q.get("query"), str) or not q["query"].strip():
            raise ValueError(f"Query rỗng: {q['query_id']}")
    if dataset == "v2":
        text_set = {normalized_text(d["caption"]) for d in corpus}
        if len({normalized_text(q["query"]) for q in queries}) != len(queries):
            raise ValueError("Dataset V2 có query text trùng nhau")
        for q in queries:
            if source_name(q) != "source_a" or len(q["relevant_ids"]) != 1:
                raise ValueError(f"Dataset V2 cần đúng 1 relevant GIF/Source A: {q['query_id']}")
            if normalized_text(q["query"]) in text_set:
                raise ValueError(f"Query text bị leak vào corpus: {q['query_id']}")
            if q.get("heldout_doc_id") in doc_id_set:
                raise ValueError(f"Held-out doc bị leak vào corpus: {q['query_id']}")
    return corpus, queries


class BM25:
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.gif_ids = [d["gif_id"] for d in corpus]
        self.n = len(corpus)
        self.unique_gif_count = len(set(self.gif_ids))
        self.lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, d in enumerate(corpus):
            terms = Counter(tokenize(d["caption"]))
            self.lengths.append(sum(terms.values()))
            for term, freq in terms.items():
                postings[term].append((doc_idx, freq))
        self.avgdl = sum(self.lengths) / self.n if self.n else 0.0
        if self.avgdl <= 0:
            raise ValueError("Corpus không có từ nào có thể index")
        self.postings = dict(postings)
        # Robertson/Sparck Jones BM25 IDF, smoothed to stay nonnegative.
        self.idf = {
            term: math.log1p((self.n - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, rows in self.postings.items()
        }
        self.norms = [k1 * (1.0 - b + b * ln / self.avgdl) for ln in self.lengths]

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * self.n
        for term, qtf in Counter(tokenize(query)).items():
            if term not in self.postings:
                continue
            weighted_idf = self.idf[term] * qtf
            for doc_idx, tf in self.postings[term]:
                scores[doc_idx] += weighted_idf * (tf * (self.k1 + 1.0)) / (
                    tf + self.norms[doc_idx]
                )
        return scores

    def search(self, query: str, relevant: set[str] | None = None) -> tuple[list[dict], int | None]:
        """Rank UNIQUE GIF IDs. A GIF's highest-scoring caption determines its rank.

        Tie-break: earliest corpus row (deterministic). When relevant is supplied,
        continue past Top-5 to obtain the EXACT ground-truth GIF rank for MRR.
        """
        scores = self.scores(query)
        order = sorted(range(self.n), key=lambda i: -scores[i])  # stable on equal scores
        seen: set[str] = set()
        top5: list[dict] = []
        correct_rank: int | None = None
        for doc_idx in order:
            d = self.corpus[doc_idx]
            gid = d["gif_id"]
            if gid in seen:
                continue
            seen.add(gid)
            rank = len(seen)
            if rank <= TOP_K:
                top5.append({
                    "rank": rank, "gif_id": gid, "doc_id": d["doc_id"],
                    "caption": d["caption"], "url": d.get("url", ""),
                    "score": round(scores[doc_idx], 6),
                    "correct": gid in relevant if relevant is not None else False,
                })
            if relevant is not None and gid in relevant and correct_rank is None:
                correct_rank = rank
            if len(top5) == min(TOP_K, self.unique_gif_count) and (
                relevant is None or correct_rank is not None
            ):
                break
        return top5, correct_rank


def metrics(details: list[dict]) -> dict:
    n = len(details)
    if not n:
        return {"queries": 0, "recall_at_1": None, "recall_at_5": None, "mrr": None}
    ranks = [d["rank"] for d in details]
    return {
        "queries": n,
        "recall_at_1": round(sum(r == 1 for r in ranks) / n, 4),
        "recall_at_5": round(sum(r is not None and r <= TOP_K for r in ranks) / n, 4),
        "mrr": round(sum(1.0 / r if r else 0.0 for r in ranks) / n, 4),
    }


def evaluate(index: BM25, queries: list[dict], dataset: str) -> dict:
    details = []
    for i, q in enumerate(queries, 1):
        top5, correct_rank = index.search(q["query"], set(q["relevant_ids"]))
        details.append({
            "query_id": q["query_id"], "source": source_name(q),
            "query": q["query"], "relevant_ids": q["relevant_ids"],
            "rank": correct_rank, "top5": top5,
        })
        if i % 50 == 0 or i == len(queries):
            print(f"  BM25: {i}/{len(queries)} queries")
    return {
        "model": f"BM25 (k1={index.k1}, b={index.b})", "model_key": "bm25",
        "dataset": dataset,
        "corpus_sha256": sha256(DATASETS[dataset] / "corpus.jsonl"),
        "queries_sha256": sha256(DATASETS[dataset] / "queries.jsonl"),
        "corpus_captions": index.n,
        "unique_gifs": len(set(index.gif_ids)),
        "overall": metrics(details),
        "source_a": metrics([d for d in details if d["source"] == "source_a"]),
        "source_b": metrics([d for d in details if d["source"] == "source_b"]),
        "details": details,
    }


def stats_ms(values: list[float]) -> dict:
    values = sorted(values)
    def percentile(p: float) -> float:
        index = (len(values) - 1) * p
        a = math.floor(index)
        upper = math.ceil(index)
        return values[a] + (values[upper] - values[a]) * (index - a)
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(percentile(0.50), 3),
        "p95_ms": round(percentile(0.95), 3),
    }


def measure_latency(index: BM25, queries: list[dict], dataset: str) -> dict:
    """Warm index; 1 warm-up, then 200 live query->Top5 measurements."""
    index.search(queries[0]["query"])
    observations = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q["query"])
        ms = (time.perf_counter() - t0) * 1000.0
        observations.append({"query_id": q["query_id"], "search_ms": round(ms, 4),
                             "total_ms": round(ms, 4)})
    total = stats_ms([r["total_ms"] for r in observations])
    return {
        "model": "bm25", "dataset": dataset, "device": "cpu",
        "system": platform.platform(), "queries_measured": len(queries),
        "warmup_queries": 1, "sampling": "all_queries_in_dataset_order",
        "search_top_k": TOP_K, "unit": "milliseconds_per_query",
        "excludes": ["corpus_loading", "index_building", "warmup"],
        "includes": ["query_tokenization", "BM25_scoring", "GIF_deduplication", "Top5_ranking"],
        "embedding": {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0},
        "search": total, "total": total, "details": observations,
    }


def saved_semantic_reports(dataset: str, corpus: list[dict], queries: list[dict]) -> tuple[dict, dict]:
    """Read-only, optional. Do not load any model or make any vector/API call."""
    folder = ROOT / "benchmark" / "results" / "dataset1" / "semantic"
    all_query_texts = {q["query_id"]: q["query"] for q in queries}
    result = {}
    latencies = {}
    gif_count = len({d["gif_id"] for d in corpus})
    for model in MODEL_ORDER[1:]:
        path = folder / f"{model}_results.json"
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            details = record["details"]
            byid = {d["query_id"]: d for d in details}
            if (record.get("dataset") != dataset or record.get("model_key") != model
                    or record.get("corpus_captions") != len(corpus)
                    or record.get("unique_gifs") != gif_count
                    or len(details) != len(queries) or len(byid) != len(queries)
                    or (record.get("corpus_sha256") and record["corpus_sha256"] != sha256(DATASETS[dataset] / "corpus.jsonl"))
                    or (record.get("queries_sha256") and record["queries_sha256"] != sha256(DATASETS[dataset] / "queries.jsonl"))
                    or any(qid not in byid or byid[qid].get("query") != query
                           for qid, query in all_query_texts.items())):
                raise ValueError("Không khớp 200 query/corpus/dataset")
            record["overall"] = metrics(details)
            result[model] = record
            latency_path = folder / f"{model}_latency.json"
            if latency_path.is_file():
                lt = json.loads(latency_path.read_text(encoding="utf-8"))
                if (lt.get("queries_measured", 0) > 0
                        and all("mean_ms" in lt.get(k, {}) for k in ("embedding", "search", "total"))
                        and all(d.get("query_id") in all_query_texts for d in lt.get("details", []))):
                    latencies[model] = lt
            print(f"  Đã đọc kết quả semantic (không chạy lại): {model}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"  Bỏ qua semantic result không tương thích {path.name}: {exc}")
    return result, latencies


def answer_index(corpus: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for d in corpus:
        result[d["gif_id"]].append(d)
    return dict(result)


def build_rows(corpus: list[dict], queries: list[dict], reports: dict) -> list[dict]:
    by_gif = answer_index(corpus)
    by_model = {
        model: {d["query_id"]: d for d in report["details"]}
        for model, report in reports.items()
    }
    result = []
    for q in queries:
        answers = []
        for gid in q["relevant_ids"]:
            captions = by_gif[gid]
            answers.append({
                "gif_id": gid, "captions": [d["caption"] for d in captions],
                "url": next((d.get("url", "") for d in captions if d.get("url")), ""),
            })
        result.append({
            "query_id": q["query_id"], "query": q["query"],
            "source": source_name(q), "split": q.get("split", ""),
            "answers": answers,
            "models": {m: by_model.get(m, {}).get(q["query_id"]) for m in MODEL_ORDER},
        })
    return result


def export_wide(path: Path, rows: list[dict]) -> None:
    fields = ["query_id", "query_description", "source", "split", "answer_gif_id",
              "answer_caption_in_corpus", "answer_gif_url"]
    for model in MODEL_ORDER:
        fields += [f"{model}_status", f"{model}_correct_rank", f"{model}_top1_correct",
                   f"{model}_top5_correct"]
        for rank in range(1, TOP_K + 1):
            fields += [f"{model}_top{rank}_{name}" for name in
                       ("gif_id", "caption", "score", "correct", "gif_url")]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for q in rows:
            data = {
                "query_id": q["query_id"], "query_description": q["query"],
                "source": q["source"], "split": q["split"],
                "answer_gif_id": "; ".join(str(a["gif_id"]) for a in q["answers"]),
                "answer_caption_in_corpus": " | ".join(" / ".join(a["captions"]) for a in q["answers"]),
                "answer_gif_url": "; ".join(str(a["url"]) for a in q["answers"]),
            }
            for model, item in q["models"].items():
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


def export_long(path: Path, rows: list[dict]) -> None:
    fields = ["query_id", "query_description", "answer_gif_id", "answer_caption_in_corpus",
              "model", "score_type", "correct_gif_rank", "top1_correct", "top5_correct",
              "retrieved_rank", "retrieved_gif_id", "retrieved_caption", "score",
              "correct_gif", "retrieved_gif_url"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for q in rows:
            for model, item in q["models"].items():
                if not item:
                    continue
                for top in item["top5"]:
                    writer.writerow({
                        "query_id": q["query_id"], "query_description": q["query"],
                        "answer_gif_id": "; ".join(a["gif_id"] for a in q["answers"]),
                        "answer_caption_in_corpus": " | ".join(" / ".join(a["captions"]) for a in q["answers"]),
                        "model": model, "score_type": "bm25" if model == "bm25" else "cosine",
                        "correct_gif_rank": item["rank"] if item["rank"] is not None else "",
                        "top1_correct": int(item["rank"] == 1),
                        "top5_correct": int(item["rank"] is not None and item["rank"] <= TOP_K),
                        "retrieved_rank": top["rank"], "retrieved_gif_id": top["gif_id"],
                        "retrieved_caption": top["caption"], "score": top["score"],
                        "correct_gif": int(top["correct"]), "retrieved_gif_url": top.get("url", ""),
                    })


def export_summary(path: Path, reports: dict, latencies: dict, n_queries: int) -> None:
    fields = ["model", "status", "queries", "recall_at_1_pct", "recall_at_5_pct", "mrr",
              "latency_queries", "embedding_mean_ms", "search_mean_ms", "total_mean_ms",
              "total_p50_ms", "total_p95_ms"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model in MODEL_ORDER:
            report = reports.get(model)
            row = {"model": model, "status": "ready" if report else "not_ready"}
            if report:
                metric = report["overall"]
                row.update(queries=metric["queries"],
                           recall_at_1_pct=round(metric["recall_at_1"] * 100, 2),
                           recall_at_5_pct=round(metric["recall_at_5"] * 100, 2),
                           mrr=metric["mrr"])
                if model in latencies:
                    lt = latencies[model]
                    row.update(latency_queries=lt["queries_measured"],
                               embedding_mean_ms=lt.get("embedding", {}).get("mean_ms", ""),
                               search_mean_ms=lt.get("search", {}).get("mean_ms", ""),
                               total_mean_ms=lt["total"]["mean_ms"],
                               total_p50_ms=lt["total"]["p50_ms"],
                               total_p95_ms=lt["total"]["p95_ms"])
            writer.writerow(row)


def safe_url(raw: object) -> str:
    url = str(raw or "").strip()
    return url if urlsplit(url).scheme.lower() in ("http", "https") else ""


def gif_card(gif_id: str, caption: str, url: object, *, rank: int | None = None,
             score: float | None = None, correct: bool | None = None,
             score_type: str = "BM25") -> str:
    link = safe_url(url)
    media = (f'<a href="{html.escape(link, quote=True)}" target="_blank" rel="noopener noreferrer">'
             f'<img loading="lazy" src="{html.escape(link, quote=True)}" alt="GIF"></a>') if link else '<div class="no-image">Không có URL GIF</div>'
    right = ('<span class="pill good">Đúng GIF ID</span>' if correct else '<span class="pill bad">Khác GIF ID</span>') if correct is not None else ""
    sc = f'<span class="muted">{html.escape(score_type)}: {score:.4f}</span>' if score is not None else ""
    return (f'<article class="gif-card"><div class="gif-head"><b>{"Top " + str(rank) if rank is not None else "GIF đáp án"}</b>{right}{sc}</div>'
            f'{media}<div class="gif-id">ID: {html.escape(str(gif_id))}</div>'
            f'<div class="caption">{html.escape(str(caption))}</div></article>')


def export_html(path: Path, rows: list[dict], reports: dict, latencies: dict, dataset: str) -> None:
    summary_rows = []
    for model in MODEL_ORDER:
        report = reports.get(model)
        if not report:
            summary_rows.append(f'<tr><th>{html.escape(MODEL_NAMES[model])}</th><td colspan="5" class="muted">Chưa có kết quả hợp lệ</td></tr>')
            continue
        m = report["overall"]
        latency = latencies.get(model)
        latency_text = f'{latency["total"]["mean_ms"]:.3f} ms' if latency else "—"
        measured = latency["queries_measured"] if latency else "—"
        summary_rows.append(f'<tr><th>{html.escape(MODEL_NAMES[model])}</th><td>{m["recall_at_1"]:.1%}</td>'
                            f'<td>{m["recall_at_5"]:.1%}</td><td>{m["mrr"]:.4f}</td>'
                            f'<td>{latency_text}</td><td>{measured}</td></tr>')

    cards = []
    for row in rows:
        model_sections = []
        available = [d for d in row["models"].values() if d is not None]
        all_wrong = bool(available) and all(d["rank"] != 1 for d in available)
        any_wrong = any(d["rank"] != 1 for d in available)
        status = "all_wrong" if all_wrong else ("some_wrong" if any_wrong else "all_correct")
        for model in MODEL_ORDER:
            item = row["models"].get(model)
            if item is None:
                model_sections.append(f'<section class="model"><h3>{html.escape(MODEL_NAMES[model])}</h3><p class="muted">Chưa có kết quả</p></section>')
                continue
            right = item["rank"] == 1
            mark = '<span class="pill good">Top 1 đúng</span>' if right else '<span class="pill bad">Top 1 sai</span>'
            ranking = f'Thứ hạng GIF đúng: {item["rank"]}' if item["rank"] is not None else 'Không tìm thấy GIF đáp án'
            gif_results = "".join(gif_card(
                t["gif_id"], t["caption"], t.get("url", ""), rank=t["rank"],
                score=float(t["score"]), correct=bool(t["correct"]),
                score_type="BM25" if model == "bm25" else "Cosine"
            ) for t in item["top5"])
            model_sections.append(
                f'<section class="model"><header><h3>{html.escape(MODEL_NAMES[model])}</h3>{mark}</header>'
                f'<p class="muted">{html.escape(ranking)}</p>{gif_results}</section>'
            )
        answers = "".join(gif_card(a["gif_id"], " / ".join(a["captions"]), a["url"])
                          for a in row["answers"])
        lookup = html.escape(str(row["query_id"]) + " " + row["query"], quote=True)
        cards.append(
            f'<details class="query" data-status="{status}" data-search="{lookup}">'
            f'<summary><span class="qid">{html.escape(str(row["query_id"]))}</span>'
            f'<span class="qtext">{html.escape(str(row["query"]))}</span>'
            f'<span class="pill {"bad" if all_wrong else "neutral"}">{"Tất cả sai Top 1" if all_wrong else "Xem Top 5"}</span></summary>'
            f'<div class="query-body"><h3>Đáp án trong dataset</h3><div class="answers">{answers}</div>'
            f'<h3>Kết quả tìm kiếm (Top 5)</h3><div class="models">{"".join(model_sections)}</div></div></details>'
        )
    doc = '''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GIF Retrieval — BM25 vs Semantic</title>
<style>
:root{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#f4f6fa;color:#19243c}*{box-sizing:border-box}
body{max-width:1500px;margin:auto;padding:22px}h1{font-size:28px}h2{font-size:20px}h3{font-size:15px}
.panel{background:#fff;border:1px solid #dce4ef;border-radius:14px;padding:18px;margin:15px 0}
.muted{color:#6b7890;font-size:13px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:650px}
th,td{border-bottom:1px solid #dce4ef;padding:11px;text-align:left}th{white-space:nowrap}
.tools{display:flex;gap:10px;flex-wrap:wrap}.tools input,.tools select,.tools button{padding:10px;border:1px solid #c8d1e2;border-radius:9px;background:#fff;font:inherit}
.tools input{flex:1;min-width:270px}.query{margin:12px 0;background:#fff;border:1px solid #dce4ef;border-radius:13px;overflow:hidden}
.query summary{cursor:pointer;display:flex;align-items:center;flex-wrap:wrap;gap:10px;padding:15px}.qid{font-weight:800;font-size:12px;color:#536078}.qtext{flex:1;min-width:240px;font-weight:650}
.query-body{padding:16px;border-top:1px solid #dce4ef}.models{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:10px}
.model{border:1px solid #dce4ef;border-radius:10px;background:#fbfcff;padding:10px;min-width:0}.model header{display:flex;justify-content:space-between;align-items:center;gap:10px}
.answers{max-width:420px}.gif-card{background:#fff;border:1px solid #e0e6f0;border-radius:9px;padding:9px;margin:8px 0;overflow-wrap:anywhere}
.gif-head{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;font-size:12px}
.gif-card img{display:block;width:100%;height:145px;object-fit:contain;background:#f0f4fa;border-radius:7px;margin:8px 0}
.gif-id{font-size:11px;color:#536078}.caption{font-size:12px;line-height:1.5}.no-image{padding:18px;background:#f0f4fa;color:#64748b}
.pill{font-size:11px;padding:4px 8px;border-radius:12px;font-weight:750}.good{background:#dcfce7;color:#166534}.bad{background:#fee2e2;color:#991b1b}.neutral{background:#eef2f7;color:#475569}[hidden]{display:none!important}
</style></head><body>
<h1>GIF Retrieval — BM25 (Keyword) vs Semantic Search</h1>
<p class="muted">Dataset __DATASET__: 10.000 captions, 200 queries. Chấm đúng/sai theo GIF ID, xếp hạng GIF duy nhất (nhiều captions/GIF lấy score lớn nhất). BM25 score và cosine không cùng thang đo.</p>
<section class="panel"><h2>Tổng hợp R@1, R@5, MRR &amp; latency</h2><div class="scroll"><table><thead><tr><th>Phương pháp</th><th>R@1</th><th>R@5</th><th>MRR</th><th>Latency mean</th><th>Queries đo</th></tr></thead><tbody>
__SUMMARY__</tbody></table></div>
<p class="muted">BM25 latency: tokenize + tính điểm + xếp hạng Top 5, không tính build index. Semantic latency chỉ hiển thị nếu có báo cáo cũ cùng dataset; không chạy lại model. Hardware và điều kiện đo có thể khác nhau.</p></section>
<section class="panel"><h2>Chi tiết từng query (200)</h2><div class="tools"><input id="search" type="search" placeholder="Tìm query ID hoặc nội dung..."><select id="status"><option value="all">Tất cả</option><option value="all_wrong">Tất cả phương pháp có kết quả đều sai Top 1</option><option value="some_wrong">Ít nhất một phương pháp sai</option><option value="all_correct">Tất cả đúng Top 1</option></select><button id="expand">Mở tất cả</button><button id="collapse">Thu gọn</button><span id="counter"></span></div></section>
__QUERIES__
<script>
const items=[...document.querySelectorAll('details.query')],search=document.getElementById('search'),status=document.getElementById('status'),count=document.getElementById('counter');
function refresh(){let visible=0;for(const item of items){const s=item.dataset.status,v=status.value;const ok=v==='all'||v===s||(v==='some_wrong'&&s!=='all_correct');item.hidden=!(ok&&item.dataset.search.toLowerCase().includes(search.value.toLowerCase().trim()));if(!item.hidden)visible++}count.textContent=visible+' / '+items.length+' queries'}
search.addEventListener('input',refresh);status.addEventListener('change',refresh);
document.getElementById('expand').addEventListener('click',()=>items.forEach(i=>{if(!i.hidden)i.open=true}));document.getElementById('collapse').addEventListener('click',()=>items.forEach(i=>i.open=false));refresh();
</script></body></html>'''
    doc = doc.replace("__DATASET__", html.escape(dataset)).replace("__SUMMARY__", "\n".join(summary_rows))
    path.write_text(doc.replace("__QUERIES__", "\n".join(cards)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BM25 keyword retrieval on 10k GIF captions / 200 queries")
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="v2")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 (default 1.5)")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b (default 0.75)")
    args = parser.parse_args()
    if args.k1 <= 0 or not 0 <= args.b <= 1:
        parser.error("k1 phải > 0 và b phải từ 0 đến 1")

    corpus, queries = load_data(args.dataset)
    print(f"Dataset {args.dataset}: {len(corpus)} captions, "
          f"{len({d['gif_id'] for d in corpus})} unique GIFs, {len(queries)} queries")
    print("Building BM25 inverted index (no embeddings/models/API)...")
    t0 = time.perf_counter()
    index = BM25(corpus, args.k1, args.b)
    build_seconds = time.perf_counter() - t0
    print(f"Indexed {len(index.postings)} terms in {build_seconds:.2f}s")
    bm25 = evaluate(index, queries, args.dataset)
    latency = measure_latency(index, queries, args.dataset)
    bm25["latency"] = {
        "queries_measured": latency["queries_measured"],
        "embedding": latency["embedding"], "search": latency["search"],
        "total": latency["total"], "details_file": "bm25_latency.json",
    }

    reports, latencies = saved_semantic_reports(args.dataset, corpus, queries)
    reports = {"bm25": bm25, **reports}
    latencies = {"bm25": latency, **latencies}
    rows = build_rows(corpus, queries, reports)

    # Only create/write under this exact project subfolder; never modify data/.
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "bm25_results.json", bm25)
    write_json(OUT / "bm25_latency.json", latency)
    comparison = {
        "dataset": args.dataset, "corpus_captions": len(corpus), "queries": len(queries),
        "unique_gifs": len(set(index.gif_ids)),
        "input_corpus_sha256": sha256(DATASETS[args.dataset] / "corpus.jsonl"),
        "input_queries_sha256": sha256(DATASETS[args.dataset] / "queries.jsonl"),
        "bm25_config": {"k1": args.k1, "b": args.b, "tokenizer": TOKEN_PATTERN.pattern,
                         "index_build_seconds": round(build_seconds, 4),
                         "tie_break": "original_corpus_order", "ranking_unit": "unique_gif_id"},
        "models": {
            model: ({"status": "evaluated", "overall": reports[model]["overall"],
                     "latency": latencies[model]["total"] if model in latencies else None}
                    if model in reports else {"status": "not_ready"})
            for model in MODEL_ORDER
        },
    }
    write_json(OUT / "comparison.json", comparison)
    export_summary(OUT / "benchmark_summary.csv", reports, latencies, len(queries))
    export_wide(OUT / "query_comparison.csv", rows)
    export_long(OUT / "query_comparison_long.csv", rows)
    export_html(OUT / "query_comparison.html", rows, reports, latencies, args.dataset)

    m = bm25["overall"]
    print(f"\nBM25  R@1={m['recall_at_1']:.1%} | R@5={m['recall_at_5']:.1%} | "
          f"MRR={m['mrr']:.4f} | latency={latency['total']['mean_ms']:.3f} ms/query")
    print("\nALL OUTPUTS ARE IN:", OUT)
    for name in ("bm25_results.json", "bm25_latency.json", "comparison.json",
                 "benchmark_summary.csv", "query_comparison.csv",
                 "query_comparison_long.csv", "query_comparison.html"):
        print("  -", OUT / name)
    print("Semantic models: only read saved, compatible results; no model executed.")


if __name__ == "__main__":
    main()
