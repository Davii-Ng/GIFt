"""Small offline regression tests; no model or API downloads."""
from pathlib import Path
import json
import numpy as np
from benchmark.retrieval.keyword_dataset1 import BM25
from benchmark.retrieval.hybrid_dataset1 import hybrid_one
from benchmark.datasets.edge_cases import TEST_CASES
from benchmark.datasets.build_dataset1 import validate


def mini_corpus():
    return [
        {'doc_id':'1','gif_id':'g1','caption':'a dog is running'},
        {'doc_id':'2','gif_id':'g1','caption':'dog sprints fast'},
        {'doc_id':'3','gif_id':'g2','caption':'a cat sleeps'},
        {'doc_id':'4','gif_id':'g3','caption':'a horse stands'},
        {'doc_id':'5','gif_id':'g4','caption':'a bird flying'},
        {'doc_id':'6','gif_id':'g5','caption':'a child dancing'},
    ]


def test_bm25_deduplicates_gif_and_computes_full_rank():
    index=BM25(mini_corpus())
    top,rank=index.search('dog sprints', {'g1'})
    assert rank==1
    assert top[0]['gif_id']=='g1'
    assert len({r['gif_id'] for r in top})==len(top)
    assert all(1<=r['rank']<=5 for r in top)


def test_rrf_uses_full_unique_gif_ranks():
    corpus=mini_corpus();index=BM25(corpus)
    sims=np.array([0.1,0.2,0.8,0.6,0.4,0.3])
    top,rank=hybrid_one('dog sprints',corpus,index,sims,{'g1'},60)
    assert len(top)==5 and rank is not None
    assert len({x['gif_id'] for x in top})==5
    assert any(x['correct'] for x in top)


def test_edge_case_total():
    assert len(TEST_CASES)==96
    assert len({c['id'] for c in TEST_CASES})==96


def test_frozen_dataset_valid():
    root=Path(__file__).resolve().parents[2]
    corpus,queries=validate(root/'data'/'benchmark'/'dataset1')
    assert len(corpus)==10000 and len(queries)==200


def test_rebuild_dataset_from_toy_tsv(tmp_path):
    from benchmark.datasets.build_dataset1 import build
    tsv=tmp_path/'toy.tsv'
    rows=[]
    for i in range(8):
        for j in range(3):
            rows.append(f'https://example.com/{i}.gif\tunique action {i} caption {j}')
    tsv.write_text('\n'.join(rows)+'\n',encoding='utf-8')
    a=tmp_path/'run_a';b=tmp_path/'run_b'
    build(tsv,a,seed=9,docs_count=12,queries_count=3)
    build(tsv,b,seed=9,docs_count=12,queries_count=3)
    assert (a/'corpus.jsonl').read_bytes()==(b/'corpus.jsonl').read_bytes()
    assert (a/'queries.jsonl').read_bytes()==(b/'queries.jsonl').read_bytes()


def test_comparison_five_methods_files():
    from benchmark.reports.compare_dataset1 import build_report,OUT
    rows=build_report()
    assert len(rows)==5
    assert {r['key'] for r in rows}=={'minilm','qwen3','bge-m3','bm25','hybrid'}
    assert (OUT/'comparison_5_methods.csv').exists()
    assert (OUT/'comparison_5_methods.html').exists()
