# HƯỚNG DẪN CHẠY GIFt BENCHMARK — DATASET 1 & DATASET 2

> **Phiên bản áp dụng:** `GIFt-main-refactored` với sáu file trong `benchmark/retrieval/`:
> `semantic_dataset1.py`, `keyword_dataset1.py`, `hybrid_dataset1.py`,
> `semantic_dataset2.py`, `keyword_dataset2.py`, `hybrid_dataset2.py`.
>
> Hướng dẫn đã được đối chiếu với source code trong project refactored và hai bản cập nhật
> *Hybrid Dataset 1* / *Keyword + Hybrid Dataset 2*. **Không áp dụng nguyên xi** cho project cũ
> có các tên như `semanticdata1.py`, `hybridsearch.py`, `dataset1_retrieval_v2/`.
>
> **Quy tắc duy nhất:** mở PowerShell/VS Code Terminal **tại thư mục gốc chứa `app.py` và `benchmark/`**. Mọi lệnh bên dưới đều chạy ở thư mục đó. Không chạy từ `benchmark/` hoặc `retrieval/`.

---

## 0. BẮT ĐẦU Ở ĐÂU? (đọc trước khi chạy)

**Nếu mới tải project và muốn chạy lại TỪ ĐẦU:** làm **Mục 1 → Mục 2 → Mục 3 → Mục 4** theo đúng thứ tự.

**Nếu đã chạy xong BM25 + MiniLM + Qwen3 + BGE-M3 trên Dataset 1:**
- **Không** cần chạy lại `embed`, `semantic_dataset1` hoặc `keyword_dataset1`.
- Muốn thử trọng số mới: làm **Mục 3.4** (Hybrid Dataset 1), sau đó xem **Mục 3.5**.
- Chỉ muốn cập nhật bảng tổng hợp: chạy lệnh ở **Mục 3.5**.

**Nếu Dataset 2 đã có Semantic:**
- **Không** cần tải/chạy lại ba model.
- Chạy `keyword_dataset2` (nếu chưa chạy), rồi `hybrid_dataset2`, theo **Mục 4.2–4.3**.
- Chỉ muốn cập nhật báo cáo: xem **Mục 4.4**.

**Không dùng** `python -m benchmark.hybridsearch`: đây là file thừa của bản sửa cũ. Mỗi dataset chỉ dùng `benchmark/retrieval/hybrid_datasetN.py` tương ứng.

---

## 1. Chuẩn bị máy và kiểm tra project (chỉ cần làm lần đầu)

### 1.1. Mở đúng thư mục

Nếu project đang ở Downloads như trong ảnh, trong PowerShell chạy:

```powershell
cd "$env:USERPROFILE\Downloads\GIFt-main-refactored"
```

Kiểm tra:

```powershell
Get-Location
Test-Path .\benchmark\retrieval\semantic_dataset1.py
Test-Path .\benchmark\retrieval\keyword_dataset2.py
Test-Path .\benchmark\retrieval\hybrid_dataset2.py
```

Cả ba `Test-Path` phải trả về **`True`**. Nếu `False`, bạn có thể đang ở sai thư mục hoặc chưa chép đúng file cập nhật.

### 1.2. Môi trường Python

Nếu Terminal đã có `(.venv)` ở đầu dòng và chạy được `python`, có thể bỏ qua phần tạo/kích hoạt môi trường. Nếu chưa:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-benchmark.txt
```

Nếu PowerShell báo không cho activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`requirements-benchmark.txt` có tham chiếu đến `requirements.txt`; cài theo đúng lệnh trên là đủ cho phần benchmark theo cấu hình đang có (tải model lần đầu cần mạng).

### 1.3. Chạy kiểm tra code

```powershell
python -m pytest benchmark/tests -q
```

**Kỳ vọng với phiên bản đã được kiểm thử:** `6 passed` (số lượng có thể đổi nếu bạn thêm/bớt test). Không tiếp tục với các bước đắt tiền nếu test đang báo lỗi.

### 1.4. Vị trí dữ liệu và kết quả

- `benchmark/datasets/dataset1/corpus.jsonl`: corpus 10.000 captions.
- `benchmark/datasets/dataset1/queries.jsonl`: 200 queries và GIF ID đáp án.
- `data/benchmark/embeddings/dataset1/`: **cache embedding**, không phải báo cáo.
- `benchmark/datasets/edge_cases.py`: **96 edge cases Dataset 2**, không có corpus 10k GIF.
- `benchmark/results/`: **mọi CSV / JSON / HTML kết quả đánh giá**.

> Đừng đổi tên, xóa hoặc di chuyển corpus, queries, cache, hay kết quả của một dataset khi đang so sánh các lần chạy. Các bộ dữ liệu khác nhau không được ghép kết quả vào chung một benchmark.

---

## 2. BẢN ĐỒ FILE: file nào để làm gì?

| File | Vai trò | Có cần chạy trực tiếp? |
|---|---|---|
| `benchmark/datasets/build_dataset1.py` | Kiểm tra corpus 10k + 200 query; hỗ trợ tạo dataset mới | Có, trước Dataset 1 |
| `benchmark/retrieval/embed.py` | Tạo **embedding cache** cho MiniLM/Qwen3/BGE-M3 (Dataset 1) | Có, khi chưa có cache |
| `benchmark/retrieval/semantic_dataset1.py` | Đánh giá ba model trên Dataset 1 | Có |
| `benchmark/retrieval/keyword_dataset1.py` | BM25 trên Dataset 1 | Có |
| `benchmark/retrieval/hybrid_dataset1.py` | Weighted RRF BM25 + một model đã chọn trên Dataset 1 | Có, nhiều lần nếu cần |
| `benchmark/reports/compare_dataset1.py` | Tổng hợp các baseline và **tất cả** lượt Hybrid Dataset 1 | Có thể chạy lại độc lập |
| `benchmark/retrieval/semantic_dataset2.py` | Semantic ranking trên 96 edge cases | Có |
| `benchmark/retrieval/keyword_dataset2.py` | BM25 và xuất bảng comparison Dataset 2 | Có |
| `benchmark/retrieval/hybrid_dataset2.py` | Weighted RRF trên 96 edge cases Dataset 2 | Có, nhiều lần nếu cần |
| `benchmark/datasets/edge_cases.py` | Định nghĩa 96 câu / 12 nhóm Dataset 2 | **Không chạy** |
| `benchmark/__main__.py` | Các lệnh tắt `python -m benchmark ...` | Không cần chạy trực tiếp |

**Lưu ý:** `benchmark/reports/compare_dataset1.py` chỉ dùng cho Dataset 1. Báo cáo Dataset 2 được xây dựng bởi hàm `build_comparison()` nằm trong `keyword_dataset2.py`.

---

## 3. DATASET 1 — GIF Retrieval: 10.000 captions / 200 queries

### Sơ đồ đúng thứ tự

```text
[Kiểm tra Dataset 1]
         |
         +--> [BM25] -------------------------------+
         |                                          |
         +--> [Tạo embedding cache 3 model]         |
                          |                         |
                          v                         |
                  [Chạy Semantic Dataset 1] --------+
                          |                         |
                          +-----------+-------------+
                                      |
                            [Chạy Hybrid Dataset 1]
                                      |
                              [Cập nhật comparison]
```

BM25 và bước tạo embedding có thể chạy độc lập; **Hybrid cần cache embedding của model được chọn**. Để dễ theo dõi, làm theo các bước bên dưới.

### 3.1. Bước 1 — Kiểm tra Dataset 1 (KHÔNG tạo lại)

```powershell
python -m benchmark prepare
```

**Kỳ vọng:** thông báo đại ý `VALID dataset: 10000 captions, ... GIFs, 200 held-out queries` kèm hai SHA256.

**Không thêm `--rebuild`** nếu muốn giữ tương thích với dữ liệu, cache và kết quả cũ. `--rebuild` là thao tác dành cho một thí nghiệm hoàn toàn mới, yêu cầu tách riêng hoặc loại bỏ toàn bộ cache/kết quả của split cũ sau khi đã backup.

### 3.2. Bước 2 — BM25 Keyword Dataset 1

```powershell
python -m benchmark.retrieval.keyword_dataset1
```

- Không tải/gọi embedding model.
- Output: `benchmark/results/dataset1/keyword/`.
- File tiêu biểu: `bm25_results.json`, `bm25_latency.json`, `query_comparison.html`.
- Kỳ vọng xử lý đủ **200 queries** và hiển thị **R@1, R@5, MRR**.

Nếu folder keyword đã có kết quả hợp lệ và không đổi dataset/BM25 settings, **có thể bỏ qua bước này**.

### 3.3. Bước 3 — Embedding cache, sau đó Semantic Dataset 1

**Chạy ba lệnh sau khi chưa có cache đầy đủ:**

```powershell
python -m benchmark.retrieval.embed --model minilm --full
python -m benchmark.retrieval.embed --model qwen3 --full
python -m benchmark.retrieval.embed --model bge-m3 --full
```

- Dùng local models trên CPU (code hiện tại); lần đầu có thể tải model và mất thời gian.
- Mỗi model cần embedding **corpus 10.000** và **queries 200**.
- Cache được lưu trong `data/benchmark/embeddings/dataset1/`.
- Nếu đã có cache hợp lệ, script có thể tái sử dụng; không cần chạy lại khi chỉ đổi trọng số Hybrid.

**Sau khi cache đủ, chạy đánh giá Semantic:**

```powershell
python -m benchmark.retrieval.semantic_dataset1 --model all
```

Output: `benchmark/results/dataset1/semantic/`, gồm các file `minilm_results.json`, `qwen3_results.json`, `bge-m3_results.json`, `query_comparison.html` và `benchmark_summary.csv`.

Nếu **muốn đo latency Semantic mới**, dùng tùy chọn sau (sẽ chạy lại quá trình encode query để đo; không cần khi chỉ tổng hợp kết quả):

```powershell
python -m benchmark.retrieval.semantic_dataset1 --model all --measure-latency
```

> Nếu bạn đã chạy Semantic có đủ kết quả và latency thì **bỏ qua toàn bộ Mục 3.3**. Không chạy lại chỉ để cập nhật HTML/CSV tổng hợp.

### 3.4. Bước 4 — Hybrid Dataset 1 (thử nhiều model / trọng số)

Mở **duy nhất** file:

`benchmark/retrieval/hybrid_dataset1.py`

Sửa ba giá trị gần đầu file:

```python
DEFAULT_SEMANTIC_MODEL = "qwen3"   # "minilm", "qwen3", "bge-m3"
DEFAULT_SEMANTIC_WEIGHT = 0.70     # Semantic 70%; BM25 tự nhận 30%
DEFAULT_RRF_K = 60                # Phải > 0
```

Bấm **Ctrl+S**, rồi chạy:

```powershell
python -m benchmark.retrieval.hybrid_dataset1
```

- **Không** truyền `--semantic-model` hoặc `--semantic-weight` ở cuối lệnh; bản này đọc cấu hình ngay trong code.
- Dùng Weighted Reciprocal Rank Fusion (RRF) theo **rank của GIF ID**, không cộng trực tiếp cosine với BM25 score.
- Đọc cache đã tạo; **không chạy lại embedding**.
- Một lần chạy = một thư mục mới trong `benchmark/results/dataset1/hybrid/<run_id>/`.
- File của mỗi run: `hybrid_results.json`, `hybrid_latency.json`.
- Không ghi đè các run Hybrid trước.
- Sau khi chạy xong sẽ **tự cập nhật comparison Dataset 1**.

**Ví dụ ba thí nghiệm**, mỗi lần sửa code → Ctrl+S → chạy cùng một lệnh trên:

| Lần | `DEFAULT_SEMANTIC_MODEL` | `DEFAULT_SEMANTIC_WEIGHT` | BM25 tự nhận |
|---|---|---:|---:|
| 1 | `minilm` | `0.50` | 50% |
| 2 | `qwen3` | `0.70` | 30% |
| 3 | `bge-m3` | `0.30` | 70% |

### 3.5. Bước 5 — Comparison Dataset 1 (có thể chạy riêng)

Hybrid chạy thành công **đã tự cập nhật**. Nếu chỉ muốn tạo lại CSV/HTML từ các kết quả đã lưu mà không chạy model/Hybrid:

```powershell
python -m benchmark.reports.compare_dataset1
```

Hoặc lệnh tắt tương đương:

```powershell
python -m benchmark compare1
```

Xem các file:

```text
benchmark/results/dataset1/comparison/
    comparison_all_runs.csv
    comparison_all_runs.html       <-- MỞ FILE NÀY ĐỂ XEM BẢNG
    comparison_per_query.csv
```

Báo cáo gộp **BM25, MiniLM, Qwen3, BGE-M3 và mọi lượt Hybrid** đã lưu đúng dataset. Không mặc định chỉ có 5 dòng; chạy nhiều Hybrid sẽ có thêm dòng.

---

## 4. DATASET 2 — 96 EDGE CASES / 12 NHÓM

**Quan trọng:** Dataset 2 khác Dataset 1. Mỗi case có **1 positive + 2 negatives**, không tìm GIF từ corpus 10k. Metrics chính là **Hit@1, MRR** và accuracy theo **12 nhóm**. Không đối chiếu R@5 Dataset 1 với Hit@1 Dataset 2 như thể cùng bài toán.

### Sơ đồ thứ tự

```text
[Semantic Dataset 2: 3 model]
                |
                +--> [Keyword BM25 Dataset 2]
                              |
                      [Hybrid Dataset 2]
                              |
                     [Comparison Dataset 2]
```

Semantic và Keyword về kỹ thuật độc lập; nên chạy Semantic trước để Hybrid có CSV score đúng model.

### 4.1. Bước 1 — Semantic Dataset 2

**Chạy 3 model local:**

```powershell
python -m benchmark.retrieval.semantic_dataset2 --model all
```

`--model all` của file này chạy **MiniLM, Qwen3, BGE-M3** (không tự chạy Gemini).

Output: `benchmark/results/dataset2/semantic/`, trong đó tối thiểu cần có:

```text
ranking_minilm.csv
ranking_qwen3.csv
ranking_bge-m3.csv
edge_case_dashboard.html
edge_case_summary.csv
```

**Nếu đã có đủ các CSV Semantic và không muốn chạy model:** chỉ tạo lại report Semantic bằng:

```powershell
python -m benchmark.retrieval.semantic_dataset2 --report-only
```

Lệnh `--report-only` không tạo embeddings và không làm mới semantic scores.

### 4.2. Bước 2 — Keyword BM25 Dataset 2

```powershell
python -m benchmark.retrieval.keyword_dataset2
```

- Chạy BM25 độc lập trên 96 case (mỗi case có 3 câu ứng viên).
- Output: `benchmark/results/dataset2/keyword/`.
- File tiêu biểu: `ranking_bm25.csv`, `bm25_summary.csv`, `bm25_latency.json`.
- Chạy xong **tự cập nhật comparison Dataset 2**.

### 4.3. Bước 3 — Hybrid Dataset 2 (thử nhiều model / trọng số)

Mở **duy nhất** file:

`benchmark/retrieval/hybrid_dataset2.py`

Chỉnh:

```python
DEFAULT_SEMANTIC_MODEL = "qwen3"   # "minilm", "qwen3", "bge-m3"
DEFAULT_SEMANTIC_WEIGHT = 0.70     # Semantic 70%; BM25 30%
DEFAULT_RRF_K = 60
```

Bấm **Ctrl+S**, rồi chạy:

```powershell
python -m benchmark.retrieval.hybrid_dataset2
```

- Đọc `ranking_<model>.csv` từ Semantic Dataset 2 đã lưu; **không chạy lại model**.
- Tự BM25-score 3 ứng viên của từng case rồi kết hợp với semantic ranks bằng Weighted RRF.
- Lưu từng thử nghiệm tại `benchmark/results/dataset2/hybrid/<run_id>/`.
- Các file mỗi run: `hybrid_results.json`, `hybrid_latency.json`, `ranking_hybrid.csv`.
- Tự cập nhật bảng so sánh Dataset 2; không ghi đè lịch sử.

### 4.4. Bước 4 — Comparison Dataset 2 (chạy riêng nếu muốn)

Dataset 2 **không có** `benchmark/reports/compare_dataset2.py` trong phiên bản này. Phần tổng hợp nằm trong `keyword_dataset2.py` và được gọi tự động sau Keyword/Hybrid.

Nếu chỉ cần cập nhật comparison mà **không chạy lại Keyword hay Hybrid**, chạy:

```powershell
python -c "from benchmark.retrieval.keyword_dataset2 import build_comparison; build_comparison()"
```

Xem:

```text
benchmark/results/dataset2/comparison/
    comparison_all_runs.csv
    comparison_all_runs.html       <-- MỞ FILE NÀY ĐỂ XEM BẢNG
    comparison_per_query.csv
    comparison_by_category.csv    <-- ĐÚNG/96 VÀ 12 NHÓM
```

Bảng chứa **3 semantic model, BM25 và mọi lượt Hybrid** có dữ liệu hợp lệ. Dataset 2 không gộp vào bảng Dataset 1.

---

## 5. TÓM TẮT TOÀN BỘ CÁC LỆNH (copy theo thứ tự khi chạy mới)

**Lần đầu tạo môi trường:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-benchmark.txt
python -m pytest benchmark/tests -q
```

**Dataset 1 — lần đầu, trên dataset đã có:**

```powershell
python -m benchmark prepare
python -m benchmark.retrieval.keyword_dataset1
python -m benchmark.retrieval.embed --model minilm --full
python -m benchmark.retrieval.embed --model qwen3 --full
python -m benchmark.retrieval.embed --model bge-m3 --full
python -m benchmark.retrieval.semantic_dataset1 --model all
python -m benchmark.retrieval.hybrid_dataset1
python -m benchmark.reports.compare_dataset1
```

**Dataset 2 — lần đầu:**

```powershell
python -m benchmark.retrieval.semantic_dataset2 --model all
python -m benchmark.retrieval.keyword_dataset2
python -m benchmark.retrieval.hybrid_dataset2
python -c "from benchmark.retrieval.keyword_dataset2 import build_comparison; build_comparison()"
```

**Sau này, chỉ thử Hybrid với trọng số mới:**

```powershell
# Dataset 1: sửa 3 DEFAULT_* trong hybrid_dataset1.py rồi chạy
python -m benchmark.retrieval.hybrid_dataset1

# Dataset 2: sửa 3 DEFAULT_* trong hybrid_dataset2.py rồi chạy
python -m benchmark.retrieval.hybrid_dataset2
```

Không cần chạy lại các baseline khi chỉ đổi trọng số, miễn là dữ liệu và cache vẫn hợp lệ.

---

## 6. KIỂM TRA ĐÚNG/SAI SAU KHI CHẠY

Có thể kiểm tra các file chính bằng PowerShell (chạy tại root project):

```powershell
# Dataset 1
Test-Path .\benchmark\datasets\dataset1\corpus.jsonl
Test-Path .\benchmark\datasets\dataset1\queries.jsonl
Test-Path .\benchmark\results\dataset1\keyword\bm25_results.json
Test-Path .\benchmark\results\dataset1\semantic\qwen3_results.json
Test-Path .\benchmark\results\dataset1\comparison\comparison_all_runs.html

# Dataset 2
Test-Path .\benchmark\results\dataset2\semantic\ranking_qwen3.csv
Test-Path .\benchmark\results\dataset2\keyword\ranking_bm25.csv
Test-Path .\benchmark\results\dataset2\comparison\comparison_all_runs.html
```

Các kiểm tra phải trả về `True` **sau khi đã chạy các bước tương ứng**. Một file tồn tại chỉ chứng minh đã được ghi ra, chưa chứng minh metrics hợp lệ; cần kiểm tra thêm thông báo `DONE`/`SAVED` trong Terminal, và không có traceback lỗi.

Liệt kê lịch sử Hybrid:

```powershell
Get-ChildItem .\benchmark\results\dataset1\hybrid -Directory
Get-ChildItem .\benchmark\results\dataset2\hybrid -Directory
```

Mở dashboard (PowerShell):

```powershell
Invoke-Item .\benchmark\results\dataset1\comparison\comparison_all_runs.html
Invoke-Item .\benchmark\results\dataset2\comparison\comparison_all_runs.html
```

---

## 7. Những lỗi dễ gặp & hướng xử lý

| Hiện tượng | Kiểm tra / xử lý |
|---|---|
| `No module named benchmark` | Mở Terminal tại **root project**, không phải trong `benchmark/`. |
| Không thấy `corpus.jsonl` / `queries.jsonl` | Kiểm tra `benchmark/datasets/dataset1/`, sau đó dùng `python -m benchmark prepare`. Không chuyển dataset vào `results/`. |
| Hybrid Dataset 1 báo thiếu `vectors.npy` | Chạy embedding cho đúng model đã chọn, ví dụ `python -m benchmark.retrieval.embed --model qwen3 --full` (nếu chọn Qwen3), **chỉ khi cache thiếu**. |
| Hybrid Dataset 2 báo thiếu `ranking_qwen3.csv` hoặc dataset hash không hợp lệ | Chạy `python -m benchmark.retrieval.semantic_dataset2 --model qwen3` để tạo lại CSV đúng phiên bản; `--report-only` không tạo được CSV thiếu. |
| Hybrid chạy xong nhưng comparison báo WARNING | Kết quả run có thể đã lưu. Thử chạy lại lệnh so sánh riêng trong **Mục 3.5 / 4.4**; đọc traceback nếu vẫn lỗi. |
| Có nhiều lần Hybrid nhưng bảng không tăng số dòng | Kiểm tra đang mở **`comparison_all_runs.html`** của **đúng Dataset** và dùng Refresh trong trình duyệt. Kiểm tra thư mục Hybrid chứa các run JSON. |
| Model/weight mới không có tác dụng | Đảm bảo sửa **đúng `hybrid_dataset1.py` hoặc `hybrid_dataset2.py`**, đã `Ctrl+S`, chạy lệnh đúng Dataset. |
| Latency Semantic Dataset 2 hiển thị `—` ở comparison | Code report hiện chưa hiển thị batch embedding time ở cùng cột BM25/Hybrid; đó không phải cùng một phạm vi đo. Không tự điền số ước tính. |
| `pytest` lỗi vì thiếu gói | Kích hoạt `.venv` rồi cài `requirements-benchmark.txt`. |

### Không nên làm

- **Không** xóa embeddings để thử thêm một trọng số Hybrid.
- **Không** chạy `prepare --rebuild` khi đang so sánh các kết quả cũ.
- **Không** đặt đầu ra benchmark trực tiếp trong `data/`; chỉ script mới được phép tạo file vào thư mục cache embeddings.
- **Không** lấy latency batch embedding của Dataset 2, BM25 per-case và Hybrid fusion-only để kết luận model nào nhanh hơn end-to-end.
- **Không** chạy `python -m benchmark.hybridsearch` hay những lệnh `--semantic-model` từ các bản hướng dẫn cũ.

---

## 8. CHECKLIST CHẠY XONG

- [ ] Root project đúng, `.venv` đã kích hoạt, dependencies đã cài.
- [ ] `python -m pytest benchmark/tests -q` chạy qua.
- [ ] Dataset 1: đã kiểm tra 10.000 captions / 200 queries.
- [ ] Dataset 1: có BM25 `bm25_results.json`.
- [ ] Dataset 1: có `minilm_results.json`, `qwen3_results.json`, `bge-m3_results.json`.
- [ ] Dataset 1: mỗi thí nghiệm Hybrid có folder riêng, có `hybrid_results.json`.
- [ ] Dataset 1: `comparison_all_runs.html` hiển thị đủ các baseline và các lần Hybrid.
- [ ] Dataset 2: có ba `ranking_<model>.csv` từ Semantic.
- [ ] Dataset 2: có `ranking_bm25.csv` từ Keyword.
- [ ] Dataset 2: mỗi thí nghiệm Hybrid có `ranking_hybrid.csv` và `hybrid_results.json`.
- [ ] Dataset 2: `comparison_all_runs.html` hiển thị tất cả phương pháp; `comparison_by_category.csv` có thống kê theo 12 nhóm.

**Kết luận:** Khi muốn chạy lại mọi thứ, đi từ Mục 1 đến Mục 4. Khi chỉ thay trọng số, **chỉ mở đúng file Hybrid của dataset đó, sửa cấu hình, lưu, chạy đúng một lệnh** — bảng so sánh tự cập nhật.
