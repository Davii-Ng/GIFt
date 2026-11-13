# GIFt Benchmark — standard structure

**Always run commands from the `GIFt-main/` project root.**

## 1. Layout

```text
benchmark/
  __main__.py                Shared CLI: python -m benchmark <command>
  datasets/
    build_dataset1.py        Build/validate the 10k-caption corpus and 200 queries
    edge_cases.py            96 edge cases (Dataset 2)
    tgif_loader.py           Read the TGIF TSV
    examples/                Small demonstration data
    dataset1/                Committed, frozen Dataset 1: 10k captions, 200 queries
  retrieval/
    embed.py                 Generate MiniLM, Qwen3, BGE-M3, and Gemini vectors
    semantic_dataset1.py     Dataset 1 semantic retrieval
    keyword_dataset1.py      Dataset 1 BM25
    hybrid_dataset1.py       Dataset 1 BM25 + semantic RRF
    semantic_dataset2.py     Dataset 2 edge-case ranking
  reports/
    compare_dataset1.py      Compare baselines and saved hybrid runs
  tests/                     Offline tests
  results/
    dataset1/
      semantic/              MiniLM, Qwen3, BGE-M3, and optionally Gemini
      keyword/               BM25
      hybrid/                BM25 + one semantic model; Qwen3 by default
      comparison/            comparison_all_runs.csv/.html and per-query CSV
    dataset2/
      semantic/              96 edge cases
```

Dataset 1 files live only in `benchmark/datasets/dataset1/`: `corpus.jsonl`, `queries.jsonl`, and `summary.json`. Vector **caches** live in `data/benchmark/embeddings/dataset1/` and are **not benchmark reports**. All benchmark CSV, HTML, and JSON reports are written under `benchmark/results/`.

For the complete Dataset 1 and Dataset 2 workflow, see [the benchmark running guide](huongdan.md).

## 2. Start from scratch

1. Create `.venv` and install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements-benchmark.txt
   ```

2. Validate the frozen corpus and queries included with the project:

   ```powershell
   python -m benchmark prepare
   ```

   To create a genuinely **new 10k/200 split**, first rename or back up `benchmark/datasets/dataset1/`, remove the old Dataset 1 files separately without deleting the TGIF source, and clear the corresponding `data/benchmark/embeddings/dataset1/` cache and old Dataset 1 reports. Then run:

   ```powershell
   python -m benchmark prepare --rebuild
   ```

   A rebuilt dataset does **not** have the same queries or hashes as the previous V2 split. Every model must be evaluated again, and old metrics must not be compared with the new results. The script does not overwrite an existing dataset.

3. Run BM25 without embeddings:

   ```powershell
   python -m benchmark keyword1
   ```

4. Generate embeddings for the three models. The first run downloads each model from its source; Qwen3 and BGE-M3 can be slow on CPU:

   ```powershell
   python -m benchmark embed --model minilm --full
   python -m benchmark embed --model qwen3 --full
   python -m benchmark embed --model bge-m3 --full
   ```

   Embedding commands only create or reuse caches under `data/benchmark/embeddings/`. They do not write reports into `data/`.

5. Run Semantic Dataset 1, optionally measuring end-to-end latency for each query:

   ```powershell
   python -m benchmark semantic1 --model all
   python -m benchmark semantic1 --model all --measure-latency
   ```

   `--measure-latency` invokes local models again to measure query speed. Gemini API calls require explicit permission. The default run uses saved vectors only.

6. Run hybrid BM25 + semantic retrieval using RRF with `k=60` over the **complete ranking of unique GIFs**:

   ```powershell
   python -m benchmark hybrid1
   ```

   Configure `DEFAULT_SEMANTIC_MODEL`, `DEFAULT_SEMANTIC_WEIGHT`, and `DEFAULT_RRF_K` in `benchmark/retrieval/hybrid_dataset1.py` before running. This version reads configuration from the module, not `--semantic-model` or `--semantic-weight` flags. The default model is Qwen3.

   Each run exports `hybrid_results.json` and `hybrid_latency.json` and updates the comparison report. Hybrid latency measures *retrieval/ranking/fusion over existing vectors*, **not end-to-end execution**; do not compare it directly with semantic query encoding plus search time.

7. Regenerate comparison tables and HTML **without running any models**:

   ```powershell
   python -m benchmark compare1
   ```

   The main outputs under `benchmark/results/dataset1/comparison/` are `comparison_all_runs.csv`, `comparison_all_runs.html`, and `comparison_per_query.csv`. The report includes the baselines and all valid saved hybrid runs. Models that have not run or use a different dataset appear as `N/A`; the report does not fabricate results.

8. Run Dataset 2, which contains 96 edge cases and uses different evaluation criteria from Dataset 1:

   ```powershell
   python -m benchmark semantic2 --model all
   python -m benchmark semantic2 --report-only
   ```

   Dataset 2 distinguishes one positive candidate from two negatives; it is not retrieval over the 10k-GIF corpus. Keep its metrics separate from Dataset 1 results.

9. Run the tests:

   ```powershell
   python -m pytest benchmark/tests -q
   ```

## 3. Fair comparisons

- Compare Dataset 1 methods only when the corpus, query IDs/text, GIF IDs, and split match. Methods rank unique GIFs using their best-matching caption.
- `R@1`, `R@5`, and `MRR` use labeled GIF IDs and do not necessarily recognize every semantically equivalent GIF.
- BM25 `total_ms` covers query-to-retrieval time and excludes inverted-index construction. Semantic `total_ms` includes query embedding plus search but excludes model loading. Hybrid currently reports only `retrieval_fusion_ms`, which must not be presented as end-to-end latency.
- Gemini is optional, may require an API key/quota, and is excluded from the five default comparison methods.
- Opening the HTML dashboard may load GIFs from Tumblr URLs. Generating HTML does not make API calls.

## 4. Items removed during the cleanup

- `benchmark/old/`: historical files and experimental scripts outside the standard pipeline.
- `data/benchmark/old/`: earlier A/B evaluation data outside the current 200-query Dataset 1.
- Legacy `data/benchmark/eval.jsonl` and top-level `corpus.jsonl`: unused by the new pipeline.
- `data/benchmark/dataset1_retrieval_v2/keyword_search/`: legacy BM25 output written into a dataset directory.
- CSV/JSON/HTML reports mixed into dataset directories were **moved** to results directories, leaving only source data and summaries in the dataset.
- `benchmark/tests/test_migrate_results.py`, which imported `benchmark.migrate_results`, was replaced with tests for the current pipeline.
- The root `test.py`, a standalone model demo rather than an automated test, was removed from the cleaned distribution. The Streamlit application was retained.

Keep the original ZIP if you need a record of the older layout. **The cleaned distribution does not modify files elsewhere on your computer.**
