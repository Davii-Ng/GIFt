"""Dataset 2 BM25 + shared comparison for semantic/keyword/historical hybrid runs.

Run from GIFt-main root:
    python -m benchmark.retrieval.keyword_dataset2

Dataset 2 consists of 96 *independent three-choice* text ranking cases (one
positive paraphrase and two hard negatives). BM25 is fitted to the three
candidate sentences in EACH case. It is NOT a 10,000-GIF retrieval benchmark.

Outputs:
  benchmark/results/dataset2/keyword/   ranking_bm25.csv, bm25_summary.csv,
                                        bm25_latency.json
  benchmark/results/dataset2/comparison/ comparison_all_runs.csv,
                                         comparison_all_runs.html,
                                         comparison_per_query.csv,
                                         comparison_by_category.csv

The comparison is also refreshed by hybrid_dataset2.py. No model, API,
embedding computation, or Dataset 1 data is required to run BM25.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import html
import json
import math
from pathlib import Path
import statistics
import time

from benchmark.datasets.edge_cases import TEST_CASES
from benchmark.retrieval.keyword_dataset1 import BM25
from benchmark.retrieval.semantic_dataset2 import dataset_hash

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmark" / "results" / "dataset2"
SEMANTIC_DIR = BASE / "semantic"
KEYWORD_DIR = BASE / "keyword"
HYBRID_DIR = BASE / "hybrid"
COMPARISON_DIR = BASE / "comparison"

# These MUST be the same values used for Dataset 1 BM25 unless benchmarking
# a specifically documented parameter change.
BM25_K1 = 1.5
BM25_B = 0.75
BASELINE_NAMES = {"minilm": "MiniLM", "qwen3": "Qwen3", "bge-m3": "BGE-M3", "gemini": "Gemini"}
LABELS = ("positive", "negative_1", "negative_2")
RESULT_COLUMNS = (
    "id", "category", "query", "positive", "negative_1", "negative_2",
    "model", "positive_score", "negative_1_score", "negative_2_score",
    "selected_label", "selected_text", "selected_score", "positive_rank",
    "hit_at_1", "reciprocal_rank", "margin", "latency_ms", "dataset_sha256",
)
SUMMARY_COLUMNS = (
    "method", "method_id", "type", "semantic_model", "semantic_weight",
    "bm25_weight", "rrf_k", "run_id", "created_at_utc", "status", "cases",
    "correct", "hit_at_1_pct", "mrr", "avg_margin", "delta_hit1_vs_semantic_pp",
    "delta_mrr_vs_semantic", "delta_hit1_vs_bm25_pp", "delta_mrr_vs_bm25",
    "latency_mean_ms", "latency_scope",
)


def validate_cases() -> None:
    if len(TEST_CASES) != 96 or len({str(c["id"]) for c in TEST_CASES}) != 96:
        raise ValueError("Dataset 2 must contain exactly 96 unique edge cases")
    for c in TEST_CASES:
        if (len(c.get("negatives", [])) != 2 or
            any(not isinstance(x, str) or not x.strip()
                for x in (c["query"], c["positive"], *c["negatives"]))):
            raise ValueError(f"Invalid three-choice case: {c.get('id')}")


def options(case: dict) -> list[str]:
    return [case["positive"], *case["negatives"]]


def rank_indices(scores: list[float]) -> tuple[list[int], list[int]]:
    """One-based ranks; ties go to negatives, like semantic_dataset2."""
    if len(scores) != 3 or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("Expected three finite scores")
    order = sorted(range(3), key=lambda i: (-float(scores[i]), -i))
    ranks = [0, 0, 0]
    for rank, index in enumerate(order, 1):
        ranks[index] = rank
    return order, ranks


def bm25_scores(case: dict) -> list[float]:
    """Use the exact Dataset 1 BM25 algorithm, over this case's 3 choices."""
    docs = [{"gif_id": f"{case['id']}-{i}", "caption": caption}
            for i, caption in enumerate(options(case))]
    engine = BM25(docs, k1=BM25_K1, b=BM25_B)
    return [float(x) for x in engine.scores(case["query"])]


def make_row(case: dict, scores: list[float], model: str,
             fingerprint: str, latency_ms: float | str = "") -> dict:
    order, ranks = rank_indices(scores)
    pos_rank = ranks[0]
    row = {
        "id": case["id"], "category": case["category"], "query": case["query"],
        "positive": case["positive"], "negative_1": case["negatives"][0],
        "negative_2": case["negatives"][1], "model": model,
        "positive_score": round(float(scores[0]), 12),
        "negative_1_score": round(float(scores[1]), 12),
        "negative_2_score": round(float(scores[2]), 12),
        "selected_label": LABELS[order[0]], "selected_text": options(case)[order[0]],
        "selected_score": round(float(scores[order[0]]), 12),
        "positive_rank": pos_rank, "hit_at_1": int(pos_rank == 1),
        "reciprocal_rank": round(1 / pos_rank, 12),
        "margin": round(float(scores[0]) - max(float(scores[1]), float(scores[2])), 12),
        "latency_ms": round(latency_ms, 6) if isinstance(latency_ms, (float, int)) else latency_ms,
        "dataset_sha256": fingerprint,
    }
    return row


def metric(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Empty ranking")
    ranks = [int(row["positive_rank"]) for row in rows]
    return {
        "cases": len(rows), "correct": sum(r == 1 for r in ranks),
        "hit_at_1_pct": round(100 * sum(r == 1 for r in ranks) / len(ranks), 4),
        "mrr": round(sum(1 / r for r in ranks) / len(ranks), 6),
        "avg_margin": round(sum(float(row["margin"]) for row in rows) / len(rows), 6),
    }


def category_metrics(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["category"])].append(r)
    return [{"category": "OVERALL", **metric(rows)}] + [
        {"category": name, **metric(group)} for name, group in sorted(groups.items())
    ]


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields) if fields else list(rows[0])
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def checked_ranking(path: Path, model: str, fp: str) -> list[dict] | None:
    """Never merge results from other datasets, missing rows or altered labels."""
    if not path.is_file():
        return None
    try:
        rows = read_csv(path)
        if len(rows) != len(TEST_CASES):
            raise ValueError("wrong number of cases")
        by_id = {str(row["id"]): row for row in rows}
        if len(by_id) != len(TEST_CASES):
            raise ValueError("duplicate case ids")
        ordered = []
        for case in TEST_CASES:
            row = by_id.get(str(case["id"]))
            if row is None:
                raise ValueError(f"missing case {case['id']}")
            for field, expected in (
                ("category", case["category"]), ("query", case["query"]),
                ("positive", case["positive"]),
                ("negative_1", case["negatives"][0]),
                ("negative_2", case["negatives"][1]),
                ("dataset_sha256", fp), ("model", model),
            ):
                if row.get(field) != str(expected):
                    raise ValueError(f"case {case['id']}: wrong {field}")
            pos_rank = int(row["positive_rank"])
            if pos_rank not in (1, 2, 3):
                raise ValueError("invalid positive rank")
            if int(row["hit_at_1"]) != int(pos_rank == 1):
                raise ValueError("hit/rank mismatch")
            if abs(float(row["reciprocal_rank"]) - (1 / pos_rank)) > 0.000002:
                raise ValueError("reciprocal rank mismatch")
            if row.get("selected_label") not in LABELS:
                raise ValueError("invalid winner")
            ordered.append(row)
        return ordered
    except (KeyError, ValueError, TypeError, OSError) as exc:
        print(f"WARNING: ignore incompatible result {path}: {exc}", flush=True)
        return None


def _baseline_row(name: str, key: str, rows: list[dict] | None) -> dict:
    if not rows:
        return {**dict.fromkeys(SUMMARY_COLUMNS, ""), "method": name,
                "method_id": key, "type": "keyword" if key == "bm25" else "semantic",
                "status": "not_run"}
    stats = metric(rows)
    times = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") not in ("", None)]
    return {
        **dict.fromkeys(SUMMARY_COLUMNS, ""),
        "method": name, "method_id": key,
        "type": "keyword" if key == "bm25" else "semantic",
        "semantic_model": key if key != "bm25" else "",
        "status": "ready", **stats,
        "latency_mean_ms": round(statistics.mean(times), 6) if len(times) == len(rows) else "",
        "latency_scope": "BM25 per-case index + scoring" if key == "bm25" else "",
    }


def _hybrid_history(fp: str) -> list[tuple[dict, list[dict]]]:
    found = []
    run_ids = set()
    paths = sorted(HYBRID_DIR.glob("*/hybrid_results.json"))
    paths += sorted(HYBRID_DIR.glob("runs/*/hybrid_results.json"))
    for path in paths:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            rid = meta["run_id"]
            model = meta["semantic_model"]
            a = float(meta["semantic_weight"])
            b = float(meta["keyword_weight"])
            k = meta["rrf_k"]
            if (rid != path.parent.name or rid in run_ids or
                meta.get("dataset_sha256") != fp or
                meta.get("dataset") != "dataset2" or
                meta.get("protocol") != "edge-3choice-weighted-rrf-v1" or
                model not in BASELINE_NAMES or not math.isfinite(a) or not math.isfinite(b) or
                not 0 <= a <= 1 or abs(a + b - 1) > 1e-8 or
                type(k) is not int or k <= 0):
                raise ValueError("incompatible hybrid metadata")
            rows = checked_ranking(path.parent / "ranking_hybrid.csv", "hybrid", fp)
            if rows is None:
                raise ValueError("incomplete hybrid rankings")
            # Verify that the metadata and CSV agree on aggregate metrics.
            actual = metric(rows)
            summary = meta["overall"]
            if (int(summary["cases"]) != actual["cases"] or
                int(summary["correct"]) != actual["correct"] or
                abs(float(summary["mrr"]) - actual["mrr"]) > 0.000002):
                raise ValueError("metadata/ranking mismatch")
            found.append((meta, rows))
            run_ids.add(rid)
        except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f"WARNING: ignore invalid hybrid run {path}: {exc}", flush=True)
    found.sort(key=lambda entry: (entry[0].get("created_at_utc", ""), entry[0]["run_id"]))
    return found


def _pct(value) -> str:
    return "—" if value in (None, "") else f"{float(value):.2f}%"


def _render_html(summaries: list[dict], detail: list[dict], categories: list[dict],
                 fingerprint: str) -> str:
    e = lambda obj: html.escape(str(obj), quote=True)
    ready = [r for r in summaries if r["status"] == "ready"]
    styles = """
:root{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#f3f6fb;color:#17243b}
*{box-sizing:border-box}body{max-width:1550px;margin:0 auto;padding:24px}
h1{margin:0 0 10px}h2{font-size:19px;margin:0 0 15px}
header{background:#14233f;color:white;padding:26px;border-radius:16px}
header p{color:#cbd5e1}.panel{background:white;border:1px solid #dce5f1;border-radius:14px;padding:20px;margin-top:18px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:10px 12px;border-bottom:1px solid #dce5f1;text-align:left;white-space:nowrap}
th{background:#f5f8ff}td.wrap{white-space:normal;min-width:230px}
.good{color:#147d45;font-weight:700}.bad{color:#b42335;font-weight:700}
.sub{color:#64748b;font-size:13px}input,select{padding:9px;border:1px solid #cbd5e1;border-radius:8px;font:inherit}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
details{margin:10px 0;border:1px solid #dce5f1;border-radius:9px;overflow:hidden}
summary{padding:14px;cursor:pointer;font-weight:600}details .inside{padding:10px 15px 16px;background:#fcfdff}
.pill{padding:3px 8px;border-radius:15px;background:#e7edf6;color:#344157;font-size:12px}
footer{margin:22px 0;font-size:12px;color:#64748b}
"""
    parts = [
        '<!doctype html><html lang="vi"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>GIFt Dataset 2 — All runs comparison</title>',
        f'<style>{styles}</style></head><body>',
        '<header><h1>Dataset 2 · Semantic / BM25 / Hybrid</h1>',
        f'<p>96 edge cases · 1 Positive + 2 Negatives mỗi câu · {len(ready)} phương pháp/lượt chạy có kết quả.</p>',
        '<p>Hit@1 và MRR là xếp hạng ba câu ứng viên, <b>không phải</b> Recall@1 hay Recall@5 trên corpus GIF.</p></header>',
        '<section class="panel"><h2>Tổng hợp các phương pháp & lịch sử Hybrid</h2>',
        '<div class="scroll"><table><thead><tr><th>Phương pháp</th><th>Trạng thái</th><th>Trọng số S/B</th><th>Đúng/96</th><th>Hit@1</th><th>MRR</th><th>Δ Hit@1 vs semantic</th><th>Δ MRR vs semantic</th><th>Latency trung bình</th></tr></thead><tbody>',
    ]
    for row in summaries:
        weights = (f'{float(row["semantic_weight"]):.0%} / {float(row["bm25_weight"]):.0%}'
                   if row["type"] == "hybrid" else "—")
        diff = row.get("delta_hit1_vs_semantic_pp", "")
        d_mrr = row.get("delta_mrr_vs_semantic", "")
        lat = f'{float(row["latency_mean_ms"]):.3f} ms' if row["latency_mean_ms"] not in ("",None) else '—'
        parts.append('<tr>' + ''.join([
            f'<td class="wrap"><strong>{e(row["method"])}</strong></td>',
            f'<td>{e(row["status"])}</td><td>{e(weights)}</td>',
            f'<td>{e(row["correct"])} / {e(row["cases"])}</td>',
            f'<td>{_pct(row["hit_at_1_pct"])}</td>',
            f'<td>{float(row["mrr"]):.4f}</td>' if row["mrr"] not in ("",None) else '<td>—</td>',
            f'<td>{float(diff):+.2f} pp</td>' if diff not in ("",None) else '<td>—</td>',
            f'<td>{float(d_mrr):+.4f}</td>' if d_mrr not in ("",None) else '<td>—</td>',
            f'<td>{e(lat)}</td>',
        ]) + '</tr>')
    parts.append('</tbody></table></div><p class="sub">Latency của các nhánh có phạm vi đo khác nhau; không dùng số này để so tốc độ end-to-end nếu chưa đo cùng giao thức.</p></section>')
    parts.append('<section class="panel"><h2>Hit@1 theo 12 nhóm</h2><div class="scroll"><table><thead><tr><th>Nhóm</th><th>Phương pháp</th><th>Đúng/tổng</th><th>Hit@1</th><th>MRR</th></tr></thead><tbody>')
    for r in categories:
        parts.append(f'<tr><td>{e(r["category"])}</td><td>{e(r["method"])}</td><td>{e(r["correct"])}/{e(r["cases"])}</td><td>{float(r["hit_at_1_pct"]):.1f}%</td><td>{float(r["mrr"]):.4f}</td></tr>')
    parts.append('</tbody></table></div></section>')
    parts.append('<section class="panel"><h2>So sánh từng query</h2><div class="controls">'
                 '<input type="search" id="search" placeholder="Tìm query, category, phương pháp...">'
                 '<select id="status"><option value="all">Tất cả</option><option value="correct">Có ít nhất một phương pháp đúng</option><option value="wrong">Tất cả phương pháp đều sai</option></select>'
                 '<span class="sub" id="counter"></span></div>')
    # List grouped by case; each comparison row carries a method/run identifier.
    by_case = defaultdict(list)
    for r in detail:
        by_case[str(r["id"])].append(r)
    for case in TEST_CASES:
        group = by_case[str(case["id"])]
        if not group:
            continue
        search = " ".join([case["query"], case["category"], *options(case),
                           *[x["method"] for x in group]]).lower()
        any_ok = any(int(x["hit_at_1"]) for x in group)
        parts.append(f'<details class="case" data-search="{e(search)}" data-hascorrect="{int(any_ok)}">'
                     f'<summary>#{e(case["id"])} · {e(case["category"])} · {e(case["query"])}</summary><div class="inside">'
                     f'<p><b>Positive:</b> {e(case["positive"])}<br><b>Negative 1:</b> {e(case["negatives"][0])}<br><b>Negative 2:</b> {e(case["negatives"][1])}</p>'
                     '<div class="scroll"><table><thead><tr><th>Phương pháp</th><th>Top 1</th><th>Positive rank</th><th>Điểm pos</th><th>Điểm neg1</th><th>Điểm neg2</th></tr></thead><tbody>')
        for d in group:
            cls = "good" if int(d["hit_at_1"]) else "bad"
            parts.append(f'<tr><td class="wrap">{e(d["method"])}</td><td class="{cls}">{e(d["selected_label"])}</td>'
                         f'<td>{e(d["positive_rank"])}</td><td>{float(d["positive_score"]):.6g}</td>'
                         f'<td>{float(d["negative_1_score"]):.6g}</td><td>{float(d["negative_2_score"]):.6g}</td></tr>')
        parts.append('</tbody></table></div></div></details>')
    parts.append(f'</section><footer>Dataset fingerprint: <code>{e(fingerprint)}</code>. '
                 'Dataset 2 chỉ đánh giá paraphrase/hard negatives được gán nhãn, không kết luận chất lượng GIF Retrieval trực tiếp. '
                 'Các score giữa semantic, BM25 và Weighted RRF không cùng thang đo.</footer>')
    parts.append('''<script>
const cards=[...document.querySelectorAll('details.case')];
const search=document.querySelector('#search'), status=document.querySelector('#status');
function update(){const needle=search.value.toLowerCase().trim();let count=0;
for(const el of cards){const ok=el.dataset.hascorrect==='1';const show=el.dataset.search.includes(needle)&&(status.value==='all'||status.value==='correct'&&ok||status.value==='wrong'&&!ok);el.hidden=!show;if(show)count++;}
document.querySelector('#counter').textContent=`${count}/${cards.length} queries`}
search.addEventListener('input',update);status.addEventListener('change',update);update();
</script></body></html>''')
    return "".join(parts)


def build_comparison() -> Path:
    """Aggregate semantic/keyword/ALL hybrid runs; callable without reranking."""
    validate_cases()
    fp = dataset_hash()
    existing: dict[str, list[dict] | None] = {}
    for model in BASELINE_NAMES:
        existing[model] = checked_ranking(SEMANTIC_DIR / f"ranking_{model}.csv", model, fp)
    existing["bm25"] = checked_ranking(KEYWORD_DIR / "ranking_bm25.csv", "bm25", fp)
    runs = _hybrid_history(fp)

    baselines = [(k, BASELINE_NAMES[k], existing[k]) for k in ("minilm", "qwen3", "bge-m3")]
    if existing["gemini"] is not None:
        baselines.append(("gemini", "Gemini", existing["gemini"]))
    baselines.append(("bm25", "BM25 (Keyword)", existing["bm25"]))
    overview = [_baseline_row(name, k, data) for k, name, data in baselines]
    by_method = [(name, k, data) for k, name, data in baselines if data]

    for meta, rows in runs:
        sem_key = meta["semantic_model"]
        alpha = float(meta["semantic_weight"])
        beta = float(meta["keyword_weight"])
        name = f'BM25 + {BASELINE_NAMES[sem_key]} (S={alpha:.0%}, B={beta:.0%}, k={meta["rrf_k"]}) · {meta["run_id"]}'
        summary = {
            **dict.fromkeys(SUMMARY_COLUMNS, ""),
            "method": name, "method_id": meta["run_id"], "type": "hybrid",
            "semantic_model": sem_key, "semantic_weight": alpha, "bm25_weight": beta,
            "rrf_k": meta["rrf_k"], "run_id": meta["run_id"],
            "created_at_utc": meta.get("created_at_utc", ""),
            "status": "ready", **metric(rows),
            "latency_mean_ms": meta.get("latency", {}).get("mean_ms", ""),
            "latency_scope": "BM25 per-case scoring + RRF fusion; semantic embedding excluded",
        }
        sm = existing.get(sem_key)
        if sm:
            m = metric(sm)
            summary["delta_hit1_vs_semantic_pp"] = round(summary["hit_at_1_pct"] - m["hit_at_1_pct"], 4)
            summary["delta_mrr_vs_semantic"] = round(summary["mrr"] - m["mrr"], 6)
        if existing["bm25"]:
            bm = metric(existing["bm25"])
            summary["delta_hit1_vs_bm25_pp"] = round(summary["hit_at_1_pct"] - bm["hit_at_1_pct"], 4)
            summary["delta_mrr_vs_bm25"] = round(summary["mrr"] - bm["mrr"], 6)
        overview.append(summary)
        by_method.append((name, meta["run_id"], rows))

    per_query = []
    per_category = []
    for method_name, method_key, rows in by_method:
        for row in rows:
            per_query.append({"method": method_name, "method_id": method_key, **row})
        for stats in category_metrics(rows):
            per_category.append({"method": method_name, "method_id": method_key, **stats})

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(COMPARISON_DIR / "comparison_all_runs.csv", overview, SUMMARY_COLUMNS)
    write_csv(COMPARISON_DIR / "comparison_per_query.csv", per_query,
              ("method", "method_id", *RESULT_COLUMNS))
    write_csv(COMPARISON_DIR / "comparison_by_category.csv", per_category,
              ("method", "method_id", "category", "cases", "correct", "hit_at_1_pct", "mrr", "avg_margin"))
    report = COMPARISON_DIR / "comparison_all_runs.html"
    tmp = report.with_name(report.name + ".tmp")
    try:
        tmp.write_text(_render_html(overview, per_query, per_category, fp), encoding="utf-8")
        tmp.replace(report)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Comparison updated: {report} ({len([r for r in overview if r['status']=='ready'])} completed methods/runs)")
    return report


def run() -> None:
    validate_cases()
    fp = dataset_hash()
    KEYWORD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Dataset 2 BM25: {len(TEST_CASES)} independent 3-choice cases (k1={BM25_K1}, b={BM25_B})", flush=True)
    for i, case in enumerate(TEST_CASES, 1):
        start = time.perf_counter()
        scores = bm25_scores(case)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        rows.append(make_row(case, scores, "bm25", fp, elapsed_ms))
        if i % 24 == 0:
            print(f"Ranked {i}/{len(TEST_CASES)} cases", flush=True)
    write_csv(KEYWORD_DIR / "ranking_bm25.csv", rows, RESULT_COLUMNS)
    summary = [{"model": "bm25", **r} for r in category_metrics(rows)]
    write_csv(KEYWORD_DIR / "bm25_summary.csv", summary)
    times = [float(r["latency_ms"]) for r in rows]
    write_json(KEYWORD_DIR / "bm25_latency.json", {
        "model": "bm25", "dataset": "dataset2", "dataset_sha256": fp,
        "cases_measured": len(times), "mean_ms": round(statistics.mean(times), 6),
        "median_ms": round(statistics.median(times), 6),
        "scope": "per-case BM25 index building + scoring for 3 candidate sentences",
        "note": "Not comparable to semantic batch embedding seconds or Dataset 1 GIF retrieval latency.",
        "measurements": [{"id": r["id"], "latency_ms": r["latency_ms"]} for r in rows],
    })
    stats = metric(rows)
    print(f"BM25: Hit@1={stats['correct']}/96 ({stats['hit_at_1_pct']:.2f}%) | MRR={stats['mrr']:.4f}")
    print(f"Saved keyword: {KEYWORD_DIR}")
    try:
        build_comparison()
    except Exception as exc:
        print(f"WARNING: BM25 saved but comparison update failed: {exc}", flush=True)
        print("Rebuild with: python -c \"from benchmark.retrieval.keyword_dataset2 import build_comparison; build_comparison()\"")


def main():
    run()


if __name__ == "__main__":
    main()
