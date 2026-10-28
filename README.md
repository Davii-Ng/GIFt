# GIFt — GIF Search + Retrieval Benchmarks

Project có **hai phần độc lập**:

1. `app.py`, `search.py`, `index.py`, `data.py`, `config.py`: demo GIF Search với Streamlit và ChromaDB (giữ gần nguyên bản).
2. `benchmark/`: benchmark GIF retrieval trên tập 10.000 captions + 200 query, bộ 96 edge cases, và so sánh BM25 / MiniLM / Qwen3 / BGE-M3 / Hybrid.

Xem **[benchmark/README.md](benchmark/README.md)** để chạy benchmark theo từng bước và xem đầy đủ bố cục folder.

## GIF Search app

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python index.py --limit 5000
streamlit run app.py
```

`data/chroma_db/` là dữ liệu của app chính, **không liên quan** đến results trong `benchmark/results/`. Bạn có thể xóa nó để tạo index mới nếu chủ động muốn.

## Benchmark nhanh

```powershell
python -m pip install -r requirements-benchmark.txt
python -m benchmark --help
python -m benchmark prepare
python -m benchmark keyword1
python -m benchmark compare1
```

Tạo corpus embeddings / chạy semantic / hybrid: xem `benchmark/README.md`.

## Quy tắc lưu file

- `data/` chứa nguồn TGIF, corpus/queries, cache vectors và Chroma DB.
- `benchmark/` chứa code đánh giá.
- `benchmark/results/` chứa **tất cả output benchmark** (CSV, JSON, HTML).
- Không đặt output benchmark trong dataset.
