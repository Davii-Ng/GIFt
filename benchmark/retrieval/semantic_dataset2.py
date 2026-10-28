"""Dataset 2: semantic edge-case benchmark + offline HTML dashboard.

This module lives at benchmark/retrieval/semantic_dataset2.py; test cases in benchmark/datasets/edge_cases.py.
From the GIFt-main root:
    python -m benchmark.retrieval.semantic_dataset2 --model all
    python -m benchmark.retrieval.semantic_dataset2 --report-only
    python -m benchmark.retrieval.semantic_dataset2 --model minilm
    python -m benchmark.retrieval.semantic_dataset2 --model gemini --delay 6

Outputs are in benchmark/results/dataset2/semantic/. The dashboard has no network
or extra dependencies, and no model/API is needed for --report-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from benchmark.datasets.edge_cases import TEST_CASES
except ImportError:
    from benchmark.datasets.edge_cases import TEST_CASES

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "dataset2" / "semantic"
MODEL_IDS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "bge-m3": "BAAI/bge-m3",
    "gemini": "gemini-embedding-001",
}
DISPLAY_NAMES = {"minilm": "MiniLM", "qwen3": "Qwen3", "bge-m3": "BGE-M3", "gemini": "Gemini"}
LOCAL_MODELS = ("minilm", "qwen3", "bge-m3")
QWEN_QUERY_INSTRUCTION = "Instruct: Retrieve GIF captions that match the user's search query.\nQuery: "
RUN_PROTOCOL = "edge-ranking-v2; query-prompt-qwen3; strict-positive-above-negatives"


def dataset_hash():
    payload = {"cases": TEST_CASES, "protocol": RUN_PROTOCOL, "models": MODEL_IDS}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize(vectors):
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix must be finite and 2-dimensional")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero-length embedding detected")
    return matrix / norms


def collect_texts():
    queries = list(dict.fromkeys(case["query"] for case in TEST_CASES))
    candidates = list(dict.fromkeys(
        text for case in TEST_CASES
        for text in [case["positive"], *case["negatives"]]
    ))
    return queries, candidates


def local_embeddings(model_name, queries, candidates, batch_size):
    from sentence_transformers import SentenceTransformer
    print(f"Loading {MODEL_IDS[model_name]} on CPU...", flush=True)
    model = SentenceTransformer(MODEL_IDS[model_name], device="cpu", trust_remote_code=False)
    query_texts = ([QWEN_QUERY_INSTRUCTION + q for q in queries]
                   if model_name == "qwen3" else queries)
    kwargs = {"prompt": ""} if model_name == "qwen3" else {}
    start = time.perf_counter()
    q = normalize(model.encode(query_texts, batch_size=batch_size, convert_to_numpy=True,
                               show_progress_bar=True, normalize_embeddings=False, **kwargs))
    c = normalize(model.encode(candidates, batch_size=batch_size, convert_to_numpy=True,
                               show_progress_bar=True, normalize_embeddings=False, **kwargs))
    elapsed = time.perf_counter() - start
    del model
    return dict(zip(queries, q)), dict(zip(candidates, c)), elapsed


def gemini_embeddings(queries, candidates, batch_size, delay, output_dir):
    from google import genai
    from google.genai import types
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY in this PowerShell session")
    client = genai.Client(api_key=key)
    cache_path = output_dir / "gemini_edge_cache.json"
    cache = {}
    if cache_path.exists():
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        if saved.get("model") == MODEL_IDS["gemini"] and saved.get("mode") == "retrieval":
            cache = saved.get("vectors", {})
    start = time.perf_counter()
    last_request = None
    for role, texts, task in (("query", queries, "RETRIEVAL_QUERY"),
                              ("document", candidates, "RETRIEVAL_DOCUMENT")):
        missing = [t for t in texts if json.dumps([role, t], ensure_ascii=False) not in cache]
        print(f"Gemini {role}: {len(texts)-len(missing)} cached, {len(missing)} missing", flush=True)
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset:offset + batch_size]
            if last_request is not None:
                time.sleep(max(0, delay - (time.monotonic() - last_request)))
            last_request = time.monotonic()
            try:
                response = client.models.embed_content(
                    model=MODEL_IDS["gemini"], contents=batch,
                    config=types.EmbedContentConfig(task_type=task),
                )
            except Exception as exc:
                raise RuntimeError("Gemini API stopped. Completed batches remain cached; "
                                   "rerun after quota resets.") from exc
            if not response.embeddings or len(response.embeddings) != len(batch):
                raise RuntimeError("Unexpected Gemini response length")
            for t, embedding in zip(batch, response.embeddings):
                cache[json.dumps([role, t], ensure_ascii=False)] = embedding.values
            cache_path.write_text(json.dumps({
                "model": MODEL_IDS["gemini"], "mode": "retrieval", "vectors": cache,
            }, ensure_ascii=False), encoding="utf-8")
            print(f"  {role}: saved {min(offset+len(batch),len(missing))}/{len(missing)} new texts", flush=True)
    q = normalize([cache[json.dumps(["query", t], ensure_ascii=False)] for t in queries])
    c = normalize([cache[json.dumps(["document", t], ensure_ascii=False)] for t in candidates])
    return dict(zip(queries, q)), dict(zip(candidates, c)), time.perf_counter() - start


def evaluate(model_name, query_vectors, candidate_vectors, fingerprint, elapsed):
    rows = []
    for case in TEST_CASES:
        options = [case["positive"], *case["negatives"]]
        scores = [float(np.dot(query_vectors[case["query"]], candidate_vectors[t])) for t in options]
        # Ties with negative are considered failures (conservative scoring).
        rank = 1 + sum(score >= scores[0] for score in scores[1:])
        selected_idx = max(range(3), key=lambda i: (scores[i], i))
        rows.append({
            "id": case["id"], "category": case["category"], "query": case["query"],
            "positive": options[0], "negative_1": options[1], "negative_2": options[2],
            "model": model_name, "positive_score": round(scores[0], 6),
            "negative_1_score": round(scores[1], 6),
            "negative_2_score": round(scores[2], 6),
            "selected_label": ["positive", "negative_1", "negative_2"][selected_idx],
            "selected_text": options[selected_idx], "selected_score": round(scores[selected_idx], 6),
            "positive_rank": rank, "hit_at_1": int(rank == 1),
            "reciprocal_rank": round(1 / rank, 6),
            "margin": round(scores[0] - max(scores[1:]), 6),
            "batch_embedding_seconds": round(elapsed, 3),
            "dataset_sha256": fingerprint,
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_valid_results(output_dir, fingerprint):
    expected_ids = {str(c["id"]) for c in TEST_CASES}
    results = {}
    for name in MODEL_IDS:
        path = output_dir / f"ranking_{name}.csv"
        if not path.exists():
            continue
        try:
            rows = read_csv(path)
            if (len(rows) != len(TEST_CASES)
                    or {r["id"] for r in rows} != expected_ids
                    or any(r.get("dataset_sha256") != fingerprint or r.get("model") != name
                           for r in rows)):
                print(f"WARNING: Ignore incompatible/stale results: {path.name}")
                continue
            results[name] = rows
        except (OSError, KeyError, ValueError) as exc:
            print(f"WARNING: Could not load {path.name}: {exc}")
    return results


def metrics(rows):
    n = len(rows)
    correct = sum(int(r["hit_at_1"]) for r in rows)
    return {"cases": n, "correct": correct, "hit_at_1_pct": round(100 * correct / n, 2),
            "mrr": round(sum(float(r["reciprocal_rank"]) for r in rows) / n, 4),
            "avg_margin": round(sum(float(r["margin"]) for r in rows) / n, 5)}


def build_reports(output_dir, fingerprint):
    results = load_valid_results(output_dir, fingerprint)
    summaries = []
    for name, rows in results.items():
        by_cat = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r)
        elapsed = float(rows[0].get("batch_embedding_seconds", 0))
        for category, subset in [("OVERALL", rows), *sorted(by_cat.items())]:
            summaries.append({"model": name, "category": category, **metrics(subset),
                              "batch_embedding_seconds": elapsed if category == "OVERALL" else ""})
    if summaries:
        write_csv(output_dir / "edge_case_summary.csv", summaries)
    else:
        (output_dir / "edge_case_summary.csv").unlink(missing_ok=True)

    long_rows = [r for name in MODEL_IDS for r in results.get(name, [])]
    if long_rows:
        write_csv(output_dir / "edge_case_details.csv", long_rows)
    else:
        (output_dir / "edge_case_details.csv").unlink(missing_ok=True)
    lookup = {name: {str(r["id"]): r for r in rows} for name, rows in results.items()}
    comparison = []
    for case in TEST_CASES:
        row = {"id": case["id"], "category": case["category"], "query": case["query"],
               "positive": case["positive"], "negative_1": case["negatives"][0],
               "negative_2": case["negatives"][1]}
        for name in MODEL_IDS:
            r = lookup.get(name, {}).get(str(case["id"]))
            for label, field in (("correct", "hit_at_1"), ("selected", "selected_text"),
                                 ("positive_score", "positive_score"),
                                 ("negative_1_score", "negative_1_score"),
                                 ("negative_2_score", "negative_2_score"),
                                 ("margin", "margin"), ("positive_rank", "positive_rank")):
                row[f"{name}_{label}"] = r.get(field, "") if r else ""
        comparison.append(row)
    write_csv(output_dir / "edge_case_comparison.csv", comparison)
    report = output_dir / "edge_case_dashboard.html"
    report.write_text(render_dashboard(results, lookup, fingerprint), encoding="utf-8")
    print(f"\nREPORT: {report}")
    print(f"SUMMARY: {output_dir / 'edge_case_summary.csv'}")
    print(f"COMPARISON: {output_dir / 'edge_case_comparison.csv'}")
    print(f"Available models: {', '.join(DISPLAY_NAMES[m] for m in results) or 'none yet'}", flush=True)


def render_dashboard(results, lookup, fingerprint):
    esc = lambda s: html.escape(str(s), quote=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(TEST_CASES)
    categories = list(dict.fromkeys(c["category"] for c in TEST_CASES))
    palette = {"minilm": "#2563eb", "qwen3": "#8b5cf6", "bge-m3": "#059669", "gemini": "#ea580c"}
    order = [m for m in MODEL_IDS if m in results]
    parts = []
    parts.append("""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Edge Case Benchmark · GIFt</title>
<style>
:root{font-family:Inter,Segoe UI,Arial,sans-serif;background:#f5f7fb;color:#182437;--bd:#e2e8f0}
*{box-sizing:border-box}body{margin:0}.wrap{max-width:1400px;margin:auto;padding:28px 22px 60px}
h1{font-size:29px;letter-spacing:-.7px;margin:3px 0 8px}h2{font-size:19px;margin:0 0 16px}
.hero{background:#111e36;color:#fff;padding:30px;border-radius:18px;margin-bottom:20px}
.hero p{color:#c6d4ea;line-height:1.6;margin:4px 0}.eyebrow{color:#9bd0ff;font-size:12px;font-weight:800;letter-spacing:2px}
.panel{background:#fff;padding:22px;border:1px solid var(--bd);border-radius:15px;margin-bottom:18px;box-shadow:0 4px 16px #1e293b05}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.metric{padding:18px;border:1px solid var(--bd);border-radius:13px;background:#fff}
.metric .name{font-weight:800;display:flex;align-items:center;gap:8px}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.number{font-size:31px;font-weight:850;margin-top:10px;letter-spacing:-1px}.small{font-size:12px;color:#66748a;line-height:1.5}
.pill{display:inline-block;padding:4px 9px;font-size:11px;border-radius:99px;font-weight:750;background:#edf2f7;color:#475569}
.pill.yes{background:#dff7e9;color:#087345}.pill.no{background:#ffeded;color:#ae3030}
.flex{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.legend{display:flex;gap:17px;flex-wrap:wrap;font-size:12px;font-weight:700;color:#546174;margin-bottom:15px}
.chartrow{display:grid;grid-template-columns:145px minmax(0,1fr) 88px;align-items:center;gap:13px;margin:11px 0;font-size:13px}
.track{height:20px;background:#eef2f7;border-radius:5px;overflow:hidden}.fill{height:100%;border-radius:5px}.chart-value{text-align:right;font-weight:800;font-variant-numeric:tabular-nums}
.table-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:11px;border-bottom:1px solid var(--bd);text-align:left;vertical-align:middle}thead th{background:#f8fafc;color:#41516a;font-weight:800;white-space:nowrap}
tbody tr:hover{background:#f8fbff}.good{color:#087345;font-weight:750}.bad{color:#bc3641;font-weight:750}.muted{color:#8b97a8}
.barcell{display:flex;align-items:center;gap:9px;min-width:125px}.barcell .track{height:9px;flex:1}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 18px}input,select{font:inherit;padding:10px 12px;border-radius:9px;border:1px solid #cbd5e1;background:#fff;color:#23334d}
input{min-width:250px;flex:1}details.case{border:1px solid var(--bd);border-radius:11px;margin-bottom:10px;overflow:hidden;background:#fff}
.case summary{list-style:none;cursor:pointer;padding:16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.case summary::-webkit-details-marker{display:none}.case summary:before{content:'+';font-size:21px;color:#64748b;font-weight:500}
.case[open] summary:before{content:'−'}.case-q{font-weight:700;flex:1;min-width:230px}.case-id{color:#6e7e95;font-size:12px;font-weight:700}
.case-body{padding:17px;border-top:1px solid var(--bd);background:#fcfdff}.options{display:grid;gap:6px;font-size:13px;padding-bottom:13px}
.model-results{display:grid;grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:11px}.model-result{border:1px solid #dbe3ee;border-radius:9px;background:#fff;padding:13px;min-width:0}
.model-result .head{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:9px}
.score-row{font-size:12px;display:flex;justify-content:space-between;gap:12px;margin:7px 0}.score-row strong{font-variant-numeric:tabular-nums}
.note{font-size:12px;color:#627086;line-height:1.6}.tag{background:#e9edf4;border-radius:4px;padding:2px 5px;font-size:11px;font-weight:700}
footer{color:#64748b;font-size:12px;line-height:1.7;margin-top:22px}@media(max-width:720px){.wrap{padding:13px}.hero{padding:20px}.chartrow{grid-template-columns:90px 1fr 72px}.panel{padding:15px}}
</style></head><body><main class="wrap">""")
    parts.append(f'<header class="hero"><div class="eyebrow">GIFt · DATASET 2</div><h1>Edge Case Benchmark</h1>'
                 f'<p>Đánh giá xếp hạng văn bản · <b>{total} câu / {len(categories)} nhóm</b> · CPU local models · Cập nhật {esc(now)}</p>'
                 '<p>Đúng = Positive có cosine similarity <b>cao hơn cả hai Negatives</b>. Kết quả không phải Recall@1 trên kho GIF.</p></header>')
    parts.append('<section class="metric-grid">')
    for name in MODEL_IDS:
        if name in results:
            m = metrics(results[name])
            detail = (f'{m["correct"]}/{total} đúng · MRR {m["mrr"]:.4f} · '
                      f'Margin {m["avg_margin"]:+.4f}')
            parts.append(f'<div class="metric"><div class="name"><span class="dot" style="background:{palette[name]}"></span>{DISPLAY_NAMES[name]}</div>'
                         f'<div class="number">{m["hit_at_1_pct"]:.1f}%</div><div class="small">{esc(detail)}</div></div>')
        else:
            parts.append(f'<div class="metric"><div class="name"><span class="dot" style="background:#cbd5e1"></span>{DISPLAY_NAMES[name]}</div>'
                         '<div class="number" style="color:#94a3b8">—</div><div class="small">Chưa chạy / chưa có số liệu hợp lệ</div></div>')
    parts.append('</section>')
    parts.append('<section class="panel" style="margin-top:18px"><h2>Accuracy tổng thể — Hit@1 (%)</h2>')
    if order:
        for name in order:
            m = metrics(results[name]); pc = m["hit_at_1_pct"]
            parts.append(f'<div class="chartrow"><b>{DISPLAY_NAMES[name]}</b><div class="track"><div class="fill" style="width:{pc}%;background:{palette[name]}"></div></div>'
                         f'<span class="chart-value">{pc:.1f}%</span></div>')
    else:
        parts.append('<p class="note">Chưa có dữ liệu. Hãy chạy <code>python -m benchmark.retrieval.semantic_dataset2 --model all</code>.</p>')
    parts.append('<p class="note">Mỗi model xếp hạng 3 câu (1 Positive và 2 Negatives). Điểm cao nhất nhận Hit@1 = 1 khi đúng Positive.</p></section>')

    parts.append('<section class="panel"><h2>Accuracy theo 12 nhóm Edge Case</h2>')
    if order:
        parts.append('<div class="legend">')
        for name in order:
            parts.append(f'<span><span class="dot" style="background:{palette[name]}"></span> {DISPLAY_NAMES[name]}</span>')
        parts.append('</div><div class="table-scroll"><table><thead><tr><th>Nhóm</th><th>Số câu</th>')
        for name in order:
            parts.append(f'<th>{DISPLAY_NAMES[name]} · đúng / tổng</th>')
        parts.append('</tr></thead><tbody>')
        for cat in categories:
            count = sum(c["category"] == cat for c in TEST_CASES)
            parts.append(f'<tr><td><b>{esc(cat)}</b></td><td>{count}</td>')
            for name in order:
                subset = [r for r in results[name] if r["category"] == cat]
                m = metrics(subset); p = m["hit_at_1_pct"]
                parts.append(f'<td><div class="barcell"><div class="track"><div class="fill" style="width:{p}%;background:{palette[name]}"></div></div>'
                             f'<b>{m["correct"]}/{count} ({p:.0f}%)</b></div></td>')
            parts.append('</tr>')
        parts.append('</tbody></table></div>')
    else:
        parts.append('<p class="note">Chưa có số liệu theo nhóm.</p>')
    parts.append('</section>')

    parts.append('<section class="panel"><h2>96 bài kiểm tra — xem đúng/sai từng model</h2>'
                 '<p class="note">Nhấp vào một câu để mở các mô tả Positive/Negative, model đã chọn câu nào và điểm cosine tương ứng.</p>'
                 '<div class="controls"><input id="search" type="search" placeholder="Tìm query, mô tả, ID..." aria-label="Tìm query">'
                 '<select id="category"><option value="">Tất cả 12 nhóm</option>')
    for cat in categories:
        parts.append(f'<option value="{esc(cat)}">{esc(cat)}</option>')
    parts.append('</select><select id="filter"><option value="all">Tất cả</option>'
                 '<option value="any_wrong">Ít nhất một model sai</option>'
                 '<option value="all_wrong">Tất cả model đã chạy đều sai</option>'
                 '<option value="all_correct">Tất cả model đã chạy đều đúng</option></select>'
                 '<span class="note" id="visible-count"></span></div><div id="cases">')
    for case in TEST_CASES:
        id_ = str(case["id"])
        present = [lookup[m][id_] for m in order]
        corrects = [int(r["hit_at_1"]) for r in present]
        stat = ("none" if not present else "all_correct" if all(corrects)
                else "all_wrong" if not any(corrects) else "mixed")
        searchable = " ".join([id_, case["query"], case["positive"], *case["negatives"]]).lower()
        parts.append(f'<details class="case" data-category="{esc(case["category"])}" data-status="{stat}" data-search="{esc(searchable)}">'
                     f'<summary><span class="case-id">#{int(id_):03d}</span><span class="case-q">{esc(case["query"])}</span>'
                     f'<span class="pill">{esc(case["category"])}</span>')
        for name in order:
            r = lookup[name][id_]
            tag = "yes" if int(r["hit_at_1"]) else "no"
            symbol = "✓" if int(r["hit_at_1"]) else "✗"
            parts.append(f'<span class="pill {tag}">{DISPLAY_NAMES[name]} {symbol}</span>')
        parts.append('</summary><div class="case-body"><div class="options">'
                     f'<div><span class="pill yes">Positive</span> {esc(case["positive"])}</div>'
                     f'<div><span class="pill no">Negative 1</span> {esc(case["negatives"][0])}</div>'
                     f'<div><span class="pill no">Negative 2</span> {esc(case["negatives"][1])}</div>'
                     '</div><div class="model-results">')
        for name in order:
            r = lookup[name][id_]; ok = int(r["hit_at_1"])
            selected = r["selected_label"].replace("negative_1", "Negative 1").replace("negative_2", "Negative 2").replace("positive", "Positive")
            parts.append(f'<div class="model-result"><div class="head"><b style="color:{palette[name]}">{DISPLAY_NAMES[name]}</b>'
                         f'<span class="pill {"yes" if ok else "no"}">{"ĐÚNG" if ok else "SAI"} · chọn {esc(selected)}</span></div>')
            for label, col in (("Positive", "positive_score"), ("Negative 1", "negative_1_score"), ("Negative 2", "negative_2_score")):
                cls = "good" if label == "Positive" else ""
                parts.append(f'<div class="score-row"><span class="{cls}">{label}</span><strong>{float(r[col]):+.4f}</strong></div>')
            margin = float(r["margin"])
            parts.append(f'<div class="score-row" style="border-top:1px solid #e2e8f0;padding-top:8px"><b>Margin (Pos - max Neg)</b>'
                         f'<strong class="{"good" if margin>0 else "bad"}">{margin:+.4f}</strong></div></div>')
        if not order:
            parts.append('<p class="note">Chưa có model nào được đánh giá.</p>')
        parts.append('</div></div></details>')
    parts.append('</div></section>')
    parts.append(f'<footer>Dữ liệu: edge_cases.py · {total} câu / {len(categories)} nhóm · protocol SHA256: <code>{fingerprint[:16]}…</code>.<br>'
                 'Batch embedding time (nếu có trong CSV) là thời gian mã hóa cả tập, không phải latency cho một query. '
                 'Đây là bài kiểm tra text-to-text, không phải đánh giá trực tiếp bằng GIF thật. '
                 'Các cặp Positive/Negative là nhãn được xây dựng thủ công, cần rà soát trường hợp mơ hồ.</footer></main>')
    parts.append("""<script>
const cards=[...document.querySelectorAll('.case')];
const search=document.getElementById('search'),category=document.getElementById('category'),filter=document.getElementById('filter');
function update(){let count=0;const q=search.value.trim().toLowerCase();for(const card of cards){
let s=card.dataset.status;let matches=(card.dataset.search.includes(q)&&(!category.value||card.dataset.category===category.value)&&
(filter.value==='all'||filter.value==='any_wrong'&&(s==='all_wrong'||s==='mixed')||filter.value==='all_wrong'&&s==='all_wrong'||filter.value==='all_correct'&&s==='all_correct'));
card.hidden=!matches;if(matches)count++;}document.getElementById('visible-count').textContent=count+' / '+cards.length+' queries';}
[search,category,filter].forEach(x=>x.addEventListener(x===search?'input':'change',update));update();
</script></body></html>""")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="96 edge cases / 12 categories + HTML dashboard")
    parser.add_argument("--model", choices=["all", *MODEL_IDS], default="all")
    parser.add_argument("--report-only", action="store_true", help="Rebuild HTML/CSV from saved model results; no embeddings")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--delay", type=float, default=6.0, help="Gemini API request interval (seconds)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.batch_size < 1 or args.delay < 0:
        parser.error("--batch-size >= 1; --delay >= 0 are required")
    if len(TEST_CASES) != 96 or len({c["id"] for c in TEST_CASES}) != 96:
        raise ValueError(f"Expected 96 unique test cases; found {len(TEST_CASES)}")
    if any(len(c["negatives"]) != 2 for c in TEST_CASES):
        raise ValueError("Every test needs exactly two negatives")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_hash()
    if args.report_only:
        build_reports(out, fingerprint)
        return
    queries, candidates = collect_texts()
    selected = LOCAL_MODELS if args.model == "all" else (args.model,)
    print(f"Dataset 2: {len(TEST_CASES)} cases; {len(queries)} queries; {len(candidates)} candidate texts", flush=True)
    for name in selected:
        print(f"\n== {DISPLAY_NAMES[name]} ==", flush=True)
        try:
            if name == "gemini":
                q, c, elapsed = gemini_embeddings(queries, candidates, args.batch_size, args.delay, out)
            else:
                q, c, elapsed = local_embeddings(name, queries, candidates, args.batch_size)
            rows = evaluate(name, q, c, fingerprint, elapsed)
            write_csv(out / f"ranking_{name}.csv", rows)
            m = metrics(rows)
            print(f"Hit@1: {m['correct']}/96 ({m['hit_at_1_pct']:.2f}%) | MRR: {m['mrr']:.4f} | "
                  f"avg margin: {m['avg_margin']:+.5f} | batch embedding: {elapsed:.2f}s", flush=True)
            build_reports(out, fingerprint)  # preserve report even if a later model fails
        except Exception:
            build_reports(out, fingerprint)
            raise
    print("\nDONE. Open edge_case_dashboard.html in your browser.", flush=True)


if __name__ == "__main__":
    main()
