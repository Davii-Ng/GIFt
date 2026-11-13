# Running the GIFt Benchmark — Dataset 1 and Dataset 2

> **Applies to:** the refactored GIFt project with these six modules under `benchmark/retrieval/`:
> `semantic_dataset1.py`, `keyword_dataset1.py`, `hybrid_dataset1.py`,
> `semantic_dataset2.py`, `keyword_dataset2.py`, and `hybrid_dataset2.py`.
>
> This guide has been checked against the refactored source and the *Hybrid Dataset 1* and
> *Keyword + Hybrid Dataset 2* updates. Do not apply it unchanged to older projects using
> names such as `semanticdata1.py`, `hybridsearch.py`, or `dataset1_retrieval_v2/`.
>
> **Working-directory rule:** open PowerShell or the VS Code terminal **at the project root containing `app.py` and `benchmark/`**. Run every command below from that directory, not from `benchmark/` or `retrieval/`.

---

## 0. Where to start

**If you have just downloaded the project and want to run everything from scratch:** complete **Section 1 → Section 2 → Section 3 → Section 4** in order.

**If BM25, MiniLM, Qwen3, and BGE-M3 have already finished on Dataset 1:**

- You do not need to rerun `embed`, `semantic_dataset1`, or `keyword_dataset1`.
- To try new weights, follow **Section 3.4** for Hybrid Dataset 1, then **Section 3.5**.
- To refresh only the comparison table, run the command in **Section 3.5**.

**If Dataset 2 semantic results already exist:**

- You do not need to download or rerun the three models.
- Run `keyword_dataset2` if needed, then `hybrid_dataset2`, following **Sections 4.2–4.3**.
- To refresh reports only, see **Section 4.4**.

Do not use `python -m benchmark.hybridsearch`; it is a leftover module name from an older revision. Each dataset uses its corresponding `benchmark/retrieval/hybrid_datasetN.py` module.

---

## 1. Prepare the environment and check the project

Complete this section once when setting up the project.

### 1.1. Open the correct directory

If the project is in Downloads, for example, run the following in PowerShell. Adjust the directory name to your local checkout:

```powershell
cd "$env:USERPROFILE\Downloads\GIFt-main-refactored"
```

Check the location and expected files:

```powershell
Get-Location
Test-Path .\benchmark\retrieval\semantic_dataset1.py
Test-Path .\benchmark\retrieval\keyword_dataset2.py
Test-Path .\benchmark\retrieval\hybrid_dataset2.py
```

All three `Test-Path` commands must return **`True`**. A `False` result usually means the terminal is in the wrong directory or the updated files have not been copied into the project.

### 1.2. Python environment

If the terminal already shows `(.venv)` and `python` works, skip environment creation and activation. Otherwise, run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-benchmark.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`requirements-benchmark.txt` includes `requirements.txt`, so the installation command above covers the benchmark's declared dependencies. Initial model downloads require an internet connection.

### 1.3. Run code checks

```powershell
python -m pytest benchmark/tests -q
```

The previously tested version reported **`6 passed`**; the count may change when tests are added or removed. Resolve test failures before starting expensive benchmark runs.

### 1.4. Data and output locations

- `benchmark/datasets/dataset1/corpus.jsonl`: the 10,000-caption corpus.
- `benchmark/datasets/dataset1/queries.jsonl`: 200 queries and their labeled GIF IDs.
- `data/benchmark/embeddings/dataset1/`: **embedding caches**, not reports.
- `benchmark/datasets/edge_cases.py`: **96 Dataset 2 edge cases**, without a 10k-GIF retrieval corpus.
- `benchmark/results/`: **all evaluation CSV, JSON, and HTML outputs**.

> Do not rename, delete, or move a dataset's corpus, queries, caches, or results while comparing runs. Results from different datasets must not be combined into the same benchmark.

---

## 2. File map

| File | Purpose | Run directly? |
|---|---|---|
| `benchmark/datasets/build_dataset1.py` | Validate the 10k-caption corpus and 200 queries; optionally create a new dataset | Yes, before Dataset 1 |
| `benchmark/retrieval/embed.py` | Generate Dataset 1 **embedding caches** for MiniLM/Qwen3/BGE-M3 | Yes, when caches are missing |
| `benchmark/retrieval/semantic_dataset1.py` | Evaluate the three models on Dataset 1 | Yes |
| `benchmark/retrieval/keyword_dataset1.py` | Evaluate BM25 on Dataset 1 | Yes |
| `benchmark/retrieval/hybrid_dataset1.py` | Weighted RRF combining BM25 with one selected model on Dataset 1 | Yes; repeat for additional experiments |
| `benchmark/reports/compare_dataset1.py` | Summarize baselines and **all** saved Dataset 1 hybrid runs | Can be rerun independently |
| `benchmark/retrieval/semantic_dataset2.py` | Semantic ranking over the 96 edge cases | Yes |
| `benchmark/retrieval/keyword_dataset2.py` | Evaluate BM25 and generate Dataset 2 comparisons | Yes |
| `benchmark/retrieval/hybrid_dataset2.py` | Weighted RRF over the 96 Dataset 2 edge cases | Yes; repeat for additional experiments |
| `benchmark/datasets/edge_cases.py` | Define the 96 cases and 12 categories for Dataset 2 | **No** |
| `benchmark/__main__.py` | Provide `python -m benchmark ...` shortcuts | No need to run the file directly |

`benchmark/reports/compare_dataset1.py` handles Dataset 1 only. Dataset 2 comparison reports are built by `build_comparison()` in `keyword_dataset2.py`.

---

## 3. Dataset 1 — GIF retrieval with 10,000 captions and 200 queries

### Execution order

```text
[Validate Dataset 1]
         |
         +--> [BM25] --------------------------------+
         |                                           |
         +--> [Build caches for three models]        |
                          |                          |
                          v                          |
                  [Run Semantic Dataset 1] -----------+
                          |                          |
                          +-----------+--------------+
                                      |
                            [Run Hybrid Dataset 1]
                                      |
                            [Refresh comparison]
```

BM25 and embedding generation can run independently. **Hybrid requires the selected model's embedding cache.** Follow the steps below for a straightforward sequence.

### 3.1. Step 1 — validate Dataset 1 without rebuilding it

```powershell
python -m benchmark prepare
```

Expected output includes a message such as `VALID dataset: 10000 captions, ... GIFs, 200 held-out queries`, followed by two SHA256 hashes.

Do not add `--rebuild` when reusing existing data, caches, and results. Rebuilding creates a separate experiment; back up the previous split and separate or remove all caches/results belonging to it.

### 3.2. Step 2 — BM25 Keyword Dataset 1

```powershell
python -m benchmark.retrieval.keyword_dataset1
```

- No embedding model is downloaded or invoked.
- Output directory: `benchmark/results/dataset1/keyword/`.
- Typical files: `bm25_results.json`, `bm25_latency.json`, and `query_comparison.html`.
- The run should process all **200 queries** and report **R@1, R@5, and MRR**.

Skip this step if valid keyword results already exist and the dataset/BM25 settings have not changed.

### 3.3. Step 3 — embedding caches, then Semantic Dataset 1

**Run these commands if complete caches are not already available:**

```powershell
python -m benchmark.retrieval.embed --model minilm --full
python -m benchmark.retrieval.embed --model qwen3 --full
python -m benchmark.retrieval.embed --model bge-m3 --full
```

- The current implementation uses local models on CPU. The first run may download models and take some time.
- Each model needs embeddings for the **10,000-caption corpus** and **200 queries**.
- Caches are saved under `data/benchmark/embeddings/dataset1/`.
- Valid caches can be reused; changing only hybrid weights does not require new embeddings.

**Once the caches are complete, run semantic evaluation:**

```powershell
python -m benchmark.retrieval.semantic_dataset1 --model all
```

Outputs under `benchmark/results/dataset1/semantic/` include `minilm_results.json`, `qwen3_results.json`, `bge-m3_results.json`, `query_comparison.html`, and `benchmark_summary.csv`.

To measure **new semantic latency**, use the following option. It encodes queries again for timing and is unnecessary when only rebuilding reports:

```powershell
python -m benchmark.retrieval.semantic_dataset1 --model all --measure-latency
```

> Skip all of Section 3.3 if complete semantic results and latency measurements already exist. Do not rerun models just to refresh summary HTML/CSV files.

### 3.4. Step 4 — Hybrid Dataset 1 with different models or weights

Open this module:

`benchmark/retrieval/hybrid_dataset1.py`

Edit the three configuration values near the top. For example:

```python
DEFAULT_SEMANTIC_MODEL = "qwen3"   # "minilm", "qwen3", or "bge-m3"
DEFAULT_SEMANTIC_WEIGHT = 0.70     # 70% semantic; BM25 automatically receives 30%
DEFAULT_RRF_K = 60                # Must be greater than zero
```

Save with **Ctrl+S**, then run:

```powershell
python -m benchmark.retrieval.hybrid_dataset1
```

- Do not append `--semantic-model` or `--semantic-weight`; this version reads configuration from the module.
- Weighted Reciprocal Rank Fusion (RRF) combines **GIF ID ranks**, rather than directly adding cosine similarity to a BM25 score.
- The run reads existing caches and **does not recompute embeddings**.
- Each experiment creates a new directory under `benchmark/results/dataset1/hybrid/<run_id>/`.
- Each run writes `hybrid_results.json` and `hybrid_latency.json`.
- Earlier hybrid runs are not overwritten.
- Completing the run **automatically refreshes the Dataset 1 comparison**.

For these three example experiments, edit the configuration, save, and rerun the same command each time:

| Run | `DEFAULT_SEMANTIC_MODEL` | `DEFAULT_SEMANTIC_WEIGHT` | Automatic BM25 weight |
|---|---|---:|---:|
| 1 | `minilm` | `0.50` | 50% |
| 2 | `qwen3` | `0.70` | 30% |
| 3 | `bge-m3` | `0.30` | 70% |

### 3.5. Step 5 — Dataset 1 comparison, independently if needed

A successful hybrid run **already refreshes the comparison**. To regenerate CSV/HTML from saved results without rerunning models or hybrid retrieval:

```powershell
python -m benchmark.reports.compare_dataset1
```

The equivalent shortcut is:

```powershell
python -m benchmark compare1
```

Inspect these files:

```text
benchmark/results/dataset1/comparison/
    comparison_all_runs.csv
    comparison_all_runs.html       <-- Open this dashboard
    comparison_per_query.csv
```

The report combines **BM25, MiniLM, Qwen3, BGE-M3, and every saved hybrid run** that matches the dataset. It is not limited to five rows; additional hybrid runs add more rows.

---

## 4. Dataset 2 — 96 edge cases across 12 categories

Dataset 2 is a different task from Dataset 1. Each case contains **one positive and two negatives**, without searching the 10k-caption corpus. Its main metrics are **Hit@1, MRR**, and accuracy across **12 categories**. Do not compare Dataset 1 R@5 directly with Dataset 2 Hit@1 as if they measured the same task.

### Execution order

```text
[Semantic Dataset 2: three models]
                |
                +--> [Keyword BM25 Dataset 2]
                              |
                      [Hybrid Dataset 2]
                              |
                    [Dataset 2 comparison]
```

Semantic and keyword evaluation are technically independent. Running semantic evaluation first makes the correct model score CSVs available for hybrid runs.

### 4.1. Step 1 — Semantic Dataset 2

**Run the three local models:**

```powershell
python -m benchmark.retrieval.semantic_dataset2 --model all
```

Here, `--model all` runs **MiniLM, Qwen3, and BGE-M3**; it does not automatically run Gemini.

The output directory, `benchmark/results/dataset2/semantic/`, should contain at least:

```text
ranking_minilm.csv
ranking_qwen3.csv
ranking_bge-m3.csv
edge_case_dashboard.html
edge_case_summary.csv
```

If the semantic CSVs already exist and you do not want to rerun models, regenerate only the semantic report:

```powershell
python -m benchmark.retrieval.semantic_dataset2 --report-only
```

`--report-only` neither creates embeddings nor refreshes semantic scores.

### 4.2. Step 2 — Keyword BM25 Dataset 2

```powershell
python -m benchmark.retrieval.keyword_dataset2
```

- Run BM25 independently on the 96 cases, each with three candidate sentences.
- Output directory: `benchmark/results/dataset2/keyword/`.
- Typical files: `ranking_bm25.csv`, `bm25_summary.csv`, and `bm25_latency.json`.
- The run **automatically refreshes the Dataset 2 comparison**.

### 4.3. Step 3 — Hybrid Dataset 2 with different models or weights

Open:

`benchmark/retrieval/hybrid_dataset2.py`

Edit the configuration, for example:

```python
DEFAULT_SEMANTIC_MODEL = "qwen3"   # "minilm", "qwen3", or "bge-m3"
DEFAULT_SEMANTIC_WEIGHT = 0.70     # 70% semantic; 30% BM25
DEFAULT_RRF_K = 60
```

Save with **Ctrl+S**, then run:

```powershell
python -m benchmark.retrieval.hybrid_dataset2
```

- Read the saved Semantic Dataset 2 `ranking_<model>.csv` without rerunning the model.
- Score each case's three candidates with BM25, then combine those ranks with semantic ranks using Weighted RRF.
- Store each experiment under `benchmark/results/dataset2/hybrid/<run_id>/`.
- Each run writes `hybrid_results.json`, `hybrid_latency.json`, and `ranking_hybrid.csv`.
- Automatically update the Dataset 2 comparison without overwriting run history.

### 4.4. Step 4 — Dataset 2 comparison, independently if needed

This version has no `benchmark/reports/compare_dataset2.py`. Dataset 2 aggregation lives in `keyword_dataset2.py` and runs automatically after keyword/hybrid evaluation.

To refresh the comparison **without rerunning keyword or hybrid evaluation**:

```powershell
python -c "from benchmark.retrieval.keyword_dataset2 import build_comparison; build_comparison()"
```

Inspect:

```text
benchmark/results/dataset2/comparison/
    comparison_all_runs.csv
    comparison_all_runs.html       <-- Open this dashboard
    comparison_per_query.csv
    comparison_by_category.csv    <-- Correct results out of 96, grouped into 12 categories
```

The table includes **three semantic models, BM25, and every valid hybrid run**. Dataset 2 is not merged into the Dataset 1 table.

---

## 5. Complete command sequence

**First-time environment setup:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-benchmark.txt
python -m pytest benchmark/tests -q
```

**Dataset 1: first run using the existing dataset:**

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

**Dataset 2: first run:**

```powershell
python -m benchmark.retrieval.semantic_dataset2 --model all
python -m benchmark.retrieval.keyword_dataset2
python -m benchmark.retrieval.hybrid_dataset2
python -c "from benchmark.retrieval.keyword_dataset2 import build_comparison; build_comparison()"
```

**Later experiments that only change hybrid weights:**

```powershell
# Dataset 1: edit the three DEFAULT_* values in hybrid_dataset1.py, then run
python -m benchmark.retrieval.hybrid_dataset1

# Dataset 2: edit the three DEFAULT_* values in hybrid_dataset2.py, then run
python -m benchmark.retrieval.hybrid_dataset2
```

Changing weights alone does not require rerunning the baselines, provided the dataset and caches remain valid.

---

## 6. Verify the outputs

Check the main files from the project root in PowerShell:

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

These checks should return `True` **after the corresponding steps have completed**. File existence alone does not prove that metrics are valid. Also check for `DONE`/`SAVED` messages in the terminal and confirm that no error traceback was printed.

List the saved hybrid runs:

```powershell
Get-ChildItem .\benchmark\results\dataset1\hybrid -Directory
Get-ChildItem .\benchmark\results\dataset2\hybrid -Directory
```

Open the dashboards in PowerShell:

```powershell
Invoke-Item .\benchmark\results\dataset1\comparison\comparison_all_runs.html
Invoke-Item .\benchmark\results\dataset2\comparison\comparison_all_runs.html
```

---

## 7. Common problems and fixes

| Symptom | Check or fix |
|---|---|
| `No module named benchmark` | Open the terminal at the **project root**, not inside `benchmark/`. |
| Missing `corpus.jsonl` or `queries.jsonl` | Check `benchmark/datasets/dataset1/`, then run `python -m benchmark prepare`. Do not move the dataset into `results/`. |
| Hybrid Dataset 1 reports missing `vectors.npy` | Generate embeddings for the selected model, such as `python -m benchmark.retrieval.embed --model qwen3 --full` for Qwen3, **only if the cache is missing**. |
| Hybrid Dataset 2 reports missing `ranking_qwen3.csv` or invalid dataset hashes | Run `python -m benchmark.retrieval.semantic_dataset2 --model qwen3` to regenerate a compatible CSV. `--report-only` cannot recreate a missing score CSV. |
| Hybrid finishes but the comparison reports a warning | The run may already be saved. Retry the standalone comparison command in **Section 3.5 or 4.4**, and inspect the traceback if it still fails. |
| More hybrid runs do not add table rows | Open **`comparison_all_runs.html` for the correct dataset** and refresh the browser. Confirm that the hybrid directory contains run JSON files. |
| Model or weight changes have no effect | Edit the correct `hybrid_dataset1.py` or `hybrid_dataset2.py`, save with **Ctrl+S**, and run the command for that dataset. |
| Dataset 2 semantic latency appears as `—` in the comparison | The report does not place batch embedding time in the same latency column as BM25/hybrid because the measurement scopes differ. Do not insert an estimated value. |
| `pytest` reports a missing package | Activate `.venv` and install `requirements-benchmark.txt`. |

### Avoid these mistakes

- Do not delete embeddings just to try another hybrid weight.
- Do not run `prepare --rebuild` while comparing existing results.
- Do not write benchmark reports directly under `data/`; only the embedding scripts should write to the embedding cache directories.
- Do not use Dataset 2 batch embedding time, BM25 per-case time, and hybrid fusion-only latency to claim an end-to-end speed comparison.
- Do not run `python -m benchmark.hybridsearch` or use the `--semantic-model` commands from older guides.

---

## 8. Completion checklist

- [ ] The terminal is at the project root, `.venv` is active, and dependencies are installed.
- [ ] `python -m pytest benchmark/tests -q` passes.
- [ ] Dataset 1: the 10,000-caption corpus and 200 queries have been validated.
- [ ] Dataset 1: BM25 `bm25_results.json` exists.
- [ ] Dataset 1: `minilm_results.json`, `qwen3_results.json`, and `bge-m3_results.json` exist.
- [ ] Dataset 1: each hybrid experiment has its own directory and `hybrid_results.json`.
- [ ] Dataset 1: `comparison_all_runs.html` shows the baselines and saved hybrid runs.
- [ ] Dataset 2: the three semantic `ranking_<model>.csv` files exist.
- [ ] Dataset 2: keyword `ranking_bm25.csv` exists.
- [ ] Dataset 2: each hybrid experiment has `ranking_hybrid.csv` and `hybrid_results.json`.
- [ ] Dataset 2: `comparison_all_runs.html` shows all valid methods, and `comparison_by_category.csv` contains statistics for the 12 categories.

For a complete rerun, work through Sections 1–4. When changing only weights, open the corresponding dataset's hybrid module, update the configuration, save, and run its command once; the comparison refreshes automatically.
