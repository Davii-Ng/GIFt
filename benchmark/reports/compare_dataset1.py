"""Compare Dataset 1's BM25/semantic baselines against ALL saved Hybrid experiments.

Run independently without any embeddings/models:
  python -m benchmark compare1
Or automatically after each: python -m benchmark.retrieval.hybrid_dataset1

Read-only inputs: dataset1 corpus/queries; semantic/keyword results;
                  hybrid/<run_id>/hybrid_results.json (also legacy hybrid/runs/*).
Outputs (only in benchmark/results/dataset1/comparison/):
  comparison_all_runs.csv     metrics + parameters for every available run
  comparison_all_runs.html    offline dashboard and all 200 queries
  comparison_per_query.csv    one row per query/method (long format)

Incompatible hashes, invalid/incomplete runs, or missing baselines are never mixed.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from benchmark.datasets.build_dataset1 import digest, validate

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmark" / "datasets" / "dataset1"
BASE = ROOT / "benchmark" / "results" / "dataset1"
OUT = BASE / "comparison"
HYBRID_ROOT = BASE / "hybrid"
BASELINES = (
    ("minilm", "MiniLM", BASE / "semantic" / "minilm_results.json"),
    ("qwen3", "Qwen3", BASE / "semantic" / "qwen3_results.json"),
    ("bge-m3", "BGE-M3", BASE / "semantic" / "bge-m3_results.json"),
    ("bm25", "BM25 (Keyword)", BASE / "keyword" / "bm25_results.json"),
)
FIELDS = [
    "method", "method_id", "type", "semantic_model", "semantic_weight", "bm25_weight",
    "rrf_k", "run_id", "created_at_utc", "status", "queries", "r_at_1", "r_at_5",
    "mrr", "delta_r1_vs_semantic", "delta_r5_vs_semantic", "delta_mrr_vs_semantic",
    "delta_mrr_vs_bm25", "latency_mean_ms", "latency_scope",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def warn(path: Path, reason: str) -> None:
    print(f"SKIPPED {path}: {reason}", flush=True)


def verified_report(path, expected_key, queries, caption_count, gif_count, hashes,
                    expected_run_id=None):
    """Reject stale results, incorrect query order/targets, duplicate IDs, bad ranks."""
    if not path.is_file():
        return None
    try:
        r = read_json(path)
        if r.get("model_key") != expected_key or r.get("dataset") != "v2":
            raise ValueError("wrong model_key/dataset")
        if r.get("corpus_captions") != caption_count or r.get("unique_gifs") != gif_count:
            raise ValueError("wrong corpus size / unique GIF count")
        if r.get("corpus_sha256") != hashes[0] or r.get("queries_sha256") != hashes[1]:
            raise ValueError("dataset SHA256 mismatch")
        details = r.get("details")
        if not isinstance(details, list) or len(details) != len(queries):
            raise ValueError("missing/incomplete query details")
        pairs = {d["query_id"]: d for d in details}
        if len(pairs) != len(details):
            raise ValueError("duplicate query IDs")
        for q in queries:
            d = pairs.get(q["query_id"])
            if d is None or d.get("query") != q["query"] or d.get("relevant_ids") != q["relevant_ids"]:
                raise ValueError("query ID/text/target differs from dataset")
            rank = d.get("rank")
            if rank is not None and (type(rank) is not int or not 1 <= rank <= gif_count):
                raise ValueError("invalid answer rank")
            top = d.get("top5")
            if not isinstance(top, list) or len(top) != min(5, gif_count):
                raise ValueError("incomplete Top-5")
            gids = [x["gif_id"] for x in top]
            if len(gids) != len(set(gids)) or [x["rank"] for x in top] != list(range(1, len(top) + 1)):
                raise ValueError("duplicate or unranked Top-5 GIFs")
            if rank is not None and rank <= 5 and gids[rank - 1] not in q["relevant_ids"]:
                raise ValueError("answer rank inconsistent with Top-5")
        # Verify reported quality metrics against the actual 200 exact GIF ranks.
        ranks = [d["rank"] for d in details]
        computed = {
            "queries": len(ranks),
            "recall_at_1": sum(x == 1 for x in ranks) / len(ranks),
            "recall_at_5": sum(x is not None and x <= 5 for x in ranks) / len(ranks),
            "mrr": sum((1.0 / x if x else 0.0) for x in ranks) / len(ranks),
        }
        existing = r.get("overall", {})
        if any((key not in existing or abs(float(existing[key]) - value) > 0.00006)
               for key, value in computed.items()):
            raise ValueError("reported metrics do not match per-query GIF ranks")
        if expected_key == "hybrid":
            if (r.get("run_id") != expected_run_id or
                r.get("protocol") != "gif-unique-weighted-rrf-v1" or
                r.get("semantic_model") not in {"minilm", "qwen3", "bge-m3"}):
                raise ValueError("incompatible/incorrect Hybrid run metadata")
            a = r.get("semantic_weight")
            b = r.get("keyword_weight")
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or (not 0 <= a <= 1) or abs(a + b - 1.0) > 1e-8:
                raise ValueError("invalid weight settings")
            if type(r.get("rrf_k")) is not int or r["rrf_k"] <= 0:
                raise ValueError("invalid rrf_k")
        return r
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        warn(path, str(exc))
        return None


def measured_latency(r, key):
    """Avoid calling Hybrid offline timings end-to-end."""
    if not r:
        return "", ""
    if key == "hybrid":
        val = r.get("latency", {}).get("mean_ms")
        return (val if val is not None else ""), "retrieval + fusion only; query encoding excluded"
    lat = r.get("latency", {})
    val = lat.get("total", {}).get("mean_ms")
    if val is not None:
        scope = ("query encoding + search (model loading excluded)" if key != "bm25"
                 else "BM25 search (index build excluded)")
        return val, scope
    # Fallback if only a standalone latency JSON exists, not a summary in results.
    rel = ("keyword" if key == "bm25" else "semantic")
    file = BASE / rel / f"{key}_latency.json"
    if file.exists():
        try:
            saved = read_json(file)
            if saved.get("queries_measured") == 200 and saved.get("total", {}).get("mean_ms") is not None:
                scope = ("BM25 search (index build excluded)" if key == "bm25"
                         else "query encoding + search (model loading excluded)")
                return saved["total"]["mean_ms"], scope
        except (ValueError, OSError, TypeError):
            pass
    return "", ""


def make_row(method_id, name, kind, report=None):
    if report is None:
        return {**dict.fromkeys(FIELDS, ""), "method_id": method_id, "method": name,
                "type": kind, "status": "not_run"}
    overall = report.get("overall", {})
    a = report.get("semantic_weight", "") if kind == "hybrid" else ""
    b = report.get("keyword_weight", "") if kind == "hybrid" else ""
    latency, latency_scope = measured_latency(report, "hybrid" if kind == "hybrid" else method_id)
    return {
        "method": name, "method_id": method_id, "type": kind,
        "semantic_model": report.get("semantic_model", "") if kind == "hybrid" else ("" if method_id == "bm25" else method_id),
        "semantic_weight": a, "bm25_weight": b,
        "rrf_k": report.get("rrf_k", "") if kind == "hybrid" else "",
        "run_id": report.get("run_id", "") if kind == "hybrid" else "",
        "created_at_utc": report.get("created_at_utc", "") if kind == "hybrid" else "",
        "status": "ready", "queries": overall.get("queries", ""),
        "r_at_1": overall.get("recall_at_1", ""),
        "r_at_5": overall.get("recall_at_5", ""), "mrr": overall.get("mrr", ""),
        "delta_r1_vs_semantic": "", "delta_r5_vs_semantic": "",
        "delta_mrr_vs_semantic": "", "delta_mrr_vs_bm25": "",
        "latency_mean_ms": latency, "latency_scope": latency_scope,
    }


def load_valid():
    corpus, queries = validate(DATA)
    hashes = (digest(DATA / "corpus.jsonl"), digest(DATA / "queries.jsonl"))
    gifs = len({d["gif_id"] for d in corpus})
    baseline = {}
    for key, _, path in BASELINES:
        rep = verified_report(path, key, queries, len(corpus), gifs, hashes)
        if rep is None:
            if not path.exists():
                print(f"NOT RUN: {path}")
        else:
            baseline[key] = rep

    history = []
    # Current layout: hybrid/<run_id>/hybrid_results.json.
    # Also support previous layout: hybrid/runs/<run_id>/hybrid_results.json.
    # This does not move or delete any existing result files.
    seen_ids = set()
    if HYBRID_ROOT.exists():
        paths = list(HYBRID_ROOT.glob("*/hybrid_results.json"))
        paths.extend(HYBRID_ROOT.glob("runs/*/hybrid_results.json"))
        for file in sorted(paths):
            rep = verified_report(file, "hybrid", queries, len(corpus), gifs, hashes,
                                  expected_run_id=file.parent.name)
            if rep is None:
                continue
            rid = rep["run_id"]
            if rid in seen_ids:
                warn(file, "duplicate run_id (already loaded)")
                continue
            seen_ids.add(rid)
            history.append(rep)
    history.sort(key=lambda rep: (rep.get("created_at_utc", ""), rep["run_id"]))
    return queries, baseline, history


def write_csv(path, fields, rows):
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_text(path, content):
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def fmt_pct(value):
    return "—" if value == "" or value is None else f"{float(value) * 100:.1f}%"


def fmt_delta(value):
    return "—" if value == "" or value is None else f"{float(value) * 100:+.1f} pp"


def esc(value):
    return html.escape(str(value))


def html_report(table, queries, report_by_id, history_count):
    best = None
    hybrid_rows = [r for r in table if r["type"] == "hybrid" and r["status"] == "ready"]
    if hybrid_rows:
        best = max(hybrid_rows, key=lambda r: (float(r["mrr"]), float(r["r_at_1"]), r["method_id"]))["method_id"]

    tbl = []
    for r in table:
        is_best = r["method_id"] == best
        label = " <span class='best'>BEST HYBRID MRR</span>" if is_best else ""
        tbl.append(
            f"<tr class='{('best-row' if is_best else '')}' data-model='{esc(r['semantic_model'])}' data-kind='{esc(r['type'])}'>"
            f"<td><strong>{esc(r['method'])}</strong>{label}<div class='sub'>{esc(r['run_id'])}</div></td>"
            f"<td>{esc(r['status'])}</td>"
            f"<td>{esc(r['semantic_model']) or '—'}</td>"
            f"<td>{fmt_pct(r['semantic_weight'])}</td><td>{fmt_pct(r['bm25_weight'])}</td>"
            f"<td>{esc(r['rrf_k']) or '—'}</td><td>{fmt_pct(r['r_at_1'])}</td>"
            f"<td>{fmt_pct(r['r_at_5'])}</td><td>{esc(r['mrr']) or '—'}</td>"
            f"<td>{fmt_delta(r['delta_r1_vs_semantic'])}</td>"
            f"<td>{fmt_delta(r['delta_mrr_vs_semantic'])}</td>"
            f"<td>{esc(r['latency_mean_ms']) or '—'}<div class='sub'>{esc(r['latency_scope'])}</div></td></tr>"
        )

    by_id = {m: {d["query_id"]: d for d in rep["details"]} for m, rep in report_by_id.items()}
    query_cards = []
    for q in queries:
        qid = q["query_id"]
        inner = []
        for row in table:
            d = by_id.get(row["method_id"], {}).get(qid)
            if d is None:
                continue
            top = d["top5"]
            first = top[0] if top else {}
            # The full Top-5 is available in comparison_per_query.csv and original JSONs.
            top_gifs = ", ".join(g["gif_id"] for g in top)
            rank = d.get("rank")
            inner.append(
                f"<tr><td>{esc(row['method'])}</td>"
                f"<td class='{('ok' if rank == 1 else '')}'>{esc(rank) if rank is not None else '—'}</td>"
                f"<td>{esc(first.get('gif_id', ''))}</td>"
                f"<td>{esc(first.get('caption', ''))}</td>"
                f"<td><small>{esc(top_gifs)}</small></td></tr>"
            )
        query_cards.append(
            f"<details class='query' data-q='{esc(qid + ' ' + q['query']).lower()}'>"
            f"<summary><strong>{esc(qid)}</strong> — {esc(q['query'])}"
            f"<span class='sub'>Target: {esc(q['relevant_ids'][0])}</span></summary>"
            f"<div class='scroll'><table><thead><tr><th>Method / Run</th><th>Target Rank</th>"
            f"<th>Top 1 GIF</th><th>Top 1 Caption</th><th>Top 5 GIF IDs</th></tr></thead>"
            f"<tbody>{''.join(inner)}</tbody></table></div></details>"
        )

    style = """<style>
    :root{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#f5f7fb;color:#152238}
    *{box-sizing:border-box}body{max-width:1600px;margin:0 auto;padding:26px}
    h1{font-size:27px}h2{font-size:19px;margin-top:26px}.note,.sub{color:#64748b}
    .note{line-height:1.7}.sub{font-size:11px;overflow-wrap:anywhere;margin-top:5px}
    .panel,.query{background:#fff;border:1px solid #dce4ed;border-radius:12px;margin:13px 0;padding:16px}
    .query{padding:0;overflow:hidden}.query summary{padding:13px;cursor:pointer}
    .query .scroll{padding:13px;border-top:1px solid #e2e8f0}
    .scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #edf1f5;vertical-align:top}
    th{white-space:nowrap;background:#f1f5fa;color:#40536d}.best-row{background:#f2f8ff}
    .best{font-size:10px;color:#1a5aab;background:#e8f0fb;border-radius:5px;padding:3px 5px}
    select,input{font:inherit;border:1px solid #bdc9d9;border-radius:8px;padding:9px 12px;background:white}
    input{min-width:min(400px,100%)}.controls{display:flex;flex-wrap:wrap;gap:10px;margin:13px 0}
    .ok{color:#15803d;font-weight:700}small{font-size:11px;overflow-wrap:anywhere}
    </style>"""
    script = """<script>
    const model=document.getElementById('model-filter');
    const search=document.getElementById('query-search');
    function filter(){
      const v=model.value;document.querySelectorAll('#summary tbody tr').forEach(r=>{
        r.hidden=!!v && r.dataset.model!==v && r.dataset.kind==='hybrid';
      });
      const word=search.value.toLowerCase().trim();let n=0;
      document.querySelectorAll('details.query').forEach(el=>{
        el.hidden=!!word && !el.dataset.q.includes(word);if(!el.hidden)n++;
      });
      document.getElementById('query-count').textContent=n+' queries';
    }
    model.addEventListener('change',filter);search.addEventListener('input',filter);filter();
    </script>"""
    return (
        '<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>GIFt — Hybrid Experiment History</title>' + style + '</head><body>'
        '<h1>GIF Retrieval · Hybrid Experiment History</h1>'
        f'<p class="note">{len(queries)} queries · 10.000 captions · '
        f'{history_count} hybrid experiments · BM25 + MiniLM/Qwen3/BGE-M3. '
        'Metrics đều tính trên unique GIF ID, cùng một dataset. '
        'Semantic weight + BM25 weight = 100%.</p>'
        '<p class="note"><b>Latency:</b> Semantic gồm query encoding + search; '
        'BM25 là search; Hybrid chỉ tính retrieval + fusion từ vector đã có. '
        '<b>Không so trực tiếp latency Hybrid với semantic end-to-end.</b> '
        'Chỉ chạy lại 2 script này không tạo embeddings mới.</p>'
        '<section class="panel"><h2>So sánh tất cả phương pháp &amp; tất cả lần chạy</h2>'
        '<div class="controls"><label>Chỉ hiện Hybrid model: <select id="model-filter">'
        '<option value="">Tất cả</option><option value="minilm">MiniLM</option>'
        '<option value="qwen3">Qwen3</option><option value="bge-m3">BGE-M3</option>'
        '</select></label></div><div class="scroll"><table id="summary"><thead><tr>'
        '<th>Method / Run</th><th>Status</th><th>Semantic</th><th>W sem</th>'
        '<th>W BM25</th><th>RRF k</th><th>R@1</th><th>R@5</th><th>MRR</th>'
        '<th>Δ R@1 vs semantic</th><th>Δ MRR vs semantic</th><th>Latency mean (ms)</th>'
        f'</tr></thead><tbody>{"".join(tbl)}</tbody></table></div></section>'
        '<h2>Chi tiết theo từng query</h2><div class="controls">'
        '<input type="search" id="query-search" placeholder="Tìm query ID / nội dung...">'
        '<span id="query-count"></span></div>'
        + "".join(query_cards) + script + '</body></html>'
    )


def build_report():
    queries, baselines, history = load_valid()
    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    all_reports = {}
    for key, name, _ in BASELINES:
        rep = baselines.get(key)
        table.append(make_row(key, name, "keyword" if key == "bm25" else "semantic", rep))
        if rep is not None:
            all_reports[key] = rep
    for r in history:
        mid = r["run_id"]
        mname = (f"BM25 + {r['semantic_model']} "
                 f"(sem {r['semantic_weight']:.0%} / BM25 {r['keyword_weight']:.0%}, k={r['rrf_k']})")
        row = make_row(mid, mname, "hybrid", r)
        semantic_rep = baselines.get(r["semantic_model"])
        if semantic_rep:
            semantic_m = semantic_rep["overall"]
            hybrid_m = r["overall"]
            row["delta_r1_vs_semantic"] = round(hybrid_m["recall_at_1"] - semantic_m["recall_at_1"], 6)
            row["delta_r5_vs_semantic"] = round(hybrid_m["recall_at_5"] - semantic_m["recall_at_5"], 6)
            row["delta_mrr_vs_semantic"] = round(hybrid_m["mrr"] - semantic_m["mrr"], 6)
        if "bm25" in baselines:
            row["delta_mrr_vs_bm25"] = round(r["overall"]["mrr"] - baselines["bm25"]["overall"]["mrr"], 6)
        table.append(row)
        all_reports[mid] = r
    write_csv(OUT / "comparison_all_runs.csv", FIELDS, table)

    detail_fields = [
        "query_id", "query", "target_gif_id", "method_id", "method", "type",
        "semantic_model", "semantic_weight", "bm25_weight", "rrf_k", "rank",
        "hit_at_1", "hit_at_5", "reciprocal_rank", "top1_gif_id", "top5_gif_ids",
    ]
    detail_rows = []
    for row in table:
        report = all_reports.get(row["method_id"])
        if report is None:
            continue
        by_q = {d["query_id"]: d for d in report["details"]}
        for q in queries:
            d = by_q[q["query_id"]]
            rank = d["rank"]
            detail_rows.append({
                "query_id": q["query_id"], "query": q["query"],
                "target_gif_id": q["relevant_ids"][0],
                "method_id": row["method_id"], "method": row["method"], "type": row["type"],
                "semantic_model": row["semantic_model"],
                "semantic_weight": row["semantic_weight"], "bm25_weight": row["bm25_weight"],
                "rrf_k": row["rrf_k"], "rank": rank if rank is not None else "",
                "hit_at_1": int(rank == 1), "hit_at_5": int(rank is not None and rank <= 5),
                "reciprocal_rank": (round(1 / rank, 8) if rank else 0),
                "top1_gif_id": d["top5"][0]["gif_id"],
                "top5_gif_ids": " | ".join(t["gif_id"] for t in d["top5"]),
            })
    write_csv(OUT / "comparison_per_query.csv", detail_fields, detail_rows)
    write_text(OUT / "comparison_all_runs.html", html_report(table, queries, all_reports, len(history)))
    print(f"UPDATED comparison: {len(history)} Hybrid run(s), {len(table)} method rows")
    print("  ", OUT / "comparison_all_runs.csv")
    print("  ", OUT / "comparison_all_runs.html")
    print("  ", OUT / "comparison_per_query.csv")
    return table


def main():
    build_report()


if __name__ == "__main__":
    main()
