# GIFt Benchmark — cấu trúc chuẩn

**Luôn chạy lệnh từ thư mục gốc `GIFt-main/`.**

## 1. Bố cục

```text
benchmark/
  __main__.py                 Lệnh chung: python -m benchmark <command>
  datasets/
    build_dataset1.py        Tạo / kiểm tra corpus 10k, queries 200
    edge_cases.py            96 edge cases (Dataset 2)
    tgif_loader.py           Đọc file TGIF TSV
    examples/                Dữ liệu demo nhỏ
    dataset1/                Dataset 1 cố định: corpus 10k, 200 queries (được commit)
  retrieval/
    embed.py                 Tạo vector theo MiniLM, Qwen3, BGE-M3, Gemini
    semantic_dataset1.py    Dataset 1 semantic retrieval
    keyword_dataset1.py     Dataset 1 BM25
    hybrid_dataset1.py      Dataset 1 BM25 + semantic RRF
    semantic_dataset2.py    Dataset 2 edge-case ranking
  reports/
    compare_dataset1.py     5 phương pháp, cùng corpus/query
  tests/                     Tests offline
  results/
    dataset1/
      semantic/              minilm, qwen3, bge-m3 (và Gemini nếu chạy)
      keyword/               BM25
      hybrid/                BM25 + 1 model semantic (mặc định Qwen3)
      comparison/            comparison_5_methods.csv + .html + per_query.csv
    dataset2/
      semantic/              96 edge cases
```

Dataset chỉ ở `benchmark/datasets/dataset1/` (`corpus.jsonl`, `queries.jsonl`, `summary.json`). Vector **cache** ở `data/benchmark/embeddings/dataset1/`, **không phải kết quả benchmark**. Bất kỳ CSV / HTML / JSON benchmark nào đều xuất vào `benchmark/results/`.

## 2. Khởi động từ đầu

1. Tạo `.venv` và cài đặt:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements-benchmark.txt
   ```

2. Kiểm tra corpus/queries đã đóng băng (đã kèm trong gói ZIP):

   ```powershell
   python -m benchmark prepare
   ```

   Nếu cần **tạo bộ 10k/200 mới thật sự**, hãy đổi tên hoặc sao lưu `benchmark/datasets/dataset1/`, xóa riêng bản dataset1 cũ (KHÔNG xóa file TGIF), xóa `data/benchmark/embeddings/dataset1/` và các kết quả Dataset 1 cũ, rồi chạy:

   ```powershell
   python -m benchmark prepare --rebuild
   ```

   Bộ dataset **tạo mới không có cùng queries/hash** với bộ V2 cũ, nên mọi model phải chạy lại; không được so sánh số liệu cũ với số liệu mới. Script không ghi đè dataset đã có.

3. Chạy BM25 không dùng embedding:

   ```powershell
   python -m benchmark keyword1
   ```

4. Tạo embeddings của 3 model. Lệnh này tải model từ nguồn tương ứng lần đầu; Qwen3 / BGE-M3 có thể chậm trên CPU:

   ```powershell
   python -m benchmark embed --model minilm --full
   python -m benchmark embed --model qwen3 --full
   python -m benchmark embed --model bge-m3 --full
   ```

   Các lệnh embedding chỉ tạo hoặc tái sử dụng cache trong `data/benchmark/embeddings/`. Không viết báo cáo vào `data/`.

5. Chạy Semantic Dataset 1, optional đo latency end-to-end từng query:

   ```powershell
   python -m benchmark semantic1 --model all
   python -m benchmark semantic1 --model all --measure-latency
   ```

   `--measure-latency` gọi lại local models để đo tốc độ query (không gọi Gemini API nếu không được cho phép). Chạy mặc định chỉ dùng vectors đã lưu.

6. Chạy Hybrid BM25 + Qwen3 theo RRF k=60 trên **toàn bộ ranking unique GIF**:

   ```powershell
   python -m benchmark hybrid1 --semantic-model qwen3
   ```

   Tự xuất `hybrid_results.json`, `hybrid_latency.json` và gọi báo cáo so sánh. Latency Hybrid chỉ đo *retrieval/ranking/fusion trên vector đã có*; **không phải end-to-end**, không nên so trực tiếp với thời gian encode+search của semantic.

7. Xuất lại bảng/HTML so sánh 5 phương pháp mà **không chạy model**:

   ```powershell
   python -m benchmark compare1
   ```

   Có **2 file chính** trong `benchmark/results/dataset1/comparison/`: `comparison_5_methods.csv` và `comparison_5_methods.html`, cùng một `comparison_per_query.csv` để đối chiếu từng query. Nếu model chưa chạy hoặc không đúng bộ dataset, hiện `N/A` (không tự tạo kết quả).

8. Dataset 2 (96 edge cases, khác cách chấm điểm Dataset 1):

   ```powershell
   python -m benchmark semantic2 --model all
   python -m benchmark semantic2 --report-only
   ```

   Dataset 2 không phải retrieval trên 10k GIF; đây là phân biệt 1 Positive với 2 Negatives. Không được gộp các metric Dataset 2 vào bảng Dataset 1.

9. Tests:

   ```powershell
   python -m pytest benchmark/tests -q
   ```

## 3. Lưu ý so sánh công bằng

- Dataset 1 chỉ so cùng corpus, query ID, query text, GIF ID, split; các phương pháp chấm unique GIF theo caption tốt nhất.
- `R@1`, `R@5`, `MRR` là dựa vào GIF ID nhãn, không đảm bảo đánh giá mọi GIF tương đương về ý nghĩa.
- BM25 `total_ms` là query->retrieval, bỏ thời gian build inverted index. Semantic `total_ms` gồm embedding query + tìm kiếm, bỏ thời gian tải model. Hybrid hiện chỉ công bố `retrieval_fusion_ms`; không hiển thị như total end-to-end.
- Gemini là tùy chọn, có thể cần API key/quota, không nằm trong 5 phương pháp so sánh mặc định.
- HTML có thể tải GIF từ URL Tumblr khi mở trình duyệt; không gọi API khi xuất HTML.

## 4. Những gì đã loại bỏ trong bản clean

- `benchmark/old/`: file lịch sử và script thử nghiệm đời cũ (không dùng trong pipeline chuẩn).
- `data/benchmark/old/`: A/B evaluation cũ, không thuộc Dataset 1 200 queries hiện hành.
- `data/benchmark/eval.jsonl`, `corpus.jsonl` cấp cũ: không dùng ở pipeline mới.
- `data/benchmark/dataset1_retrieval_v2/keyword_search/`: benchmark BM25 cũ xuất nhầm vào dataset.
- CSV/JSON/HTML nằm lẫn trong dataset được **chuyển** về results; dataset chỉ còn dữ liệu gốc và summary.
- `benchmark/tests/test_migrate_results.py`, bài test cũ import `benchmark.migrate_results`: đã thay bằng tests cho pipeline hiện hành.
- `test.py` root (demo thử model độc lập, không phải test tự động) bị bỏ khỏi bản clean; ứng dụng Streamlit vẫn giữ nguyên.

Để xem bản cũ không mất dấu, lưu ZIP ban đầu. **Bản clean không sửa file ở máy của bạn.**
