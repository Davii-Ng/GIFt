"""Build a reproducible GIF retrieval dataset: 10,000 captions / 200 held-out queries.

No API or embedding model. Frozen bundled dataset may be used as-is.
Rebuilding gives a NEW split: results and embeddings from an older split MUST NOT be mixed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'benchmark' / 'datasets' / 'dataset1'


def norm(t):
    return ' '.join(t.casefold().split())


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p, rows):
    with p.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')


def validate(folder=DEFAULT, *, expected_docs=10000, expected_queries=200):
    folder=Path(folder)
    corpus=read_jsonl(folder/'corpus.jsonl')
    queries=read_jsonl(folder/'queries.jsonl')
    if len(corpus)!=expected_docs or len(queries)!=expected_queries:
        raise ValueError(f'Expected {expected_docs} captions / {expected_queries} queries; got {len(corpus)}/{len(queries)}')
    docs={d['doc_id'] for d in corpus}
    gifs={d['gif_id'] for d in corpus}
    captions={norm(d['caption']) for d in corpus}
    if len(docs)!=len(corpus):raise ValueError('Duplicate doc_id')
    if len({q['query_id'] for q in queries})!=len(queries):raise ValueError('Duplicate query_id')
    if len({norm(q['query']) for q in queries})!=len(queries):raise ValueError('Duplicate query text')
    for q in queries:
        if len(q.get('relevant_ids',[]))!=1 or q['relevant_ids'][0] not in gifs:
            raise ValueError('Invalid label: '+q['query_id'])
        if norm(q['query']) in captions or q.get('heldout_doc_id') in docs:
            raise ValueError('Query leaked: '+q['query_id'])
    print(f'VALID dataset: {len(corpus)} captions, {len(gifs)} GIFs, {len(queries)} held-out queries')
    print('corpus SHA256:',digest(folder/'corpus.jsonl'))
    print('queries SHA256:',digest(folder/'queries.jsonl'))
    return corpus,queries


def build(tsv=None, out=DEFAULT, seed=42, docs_count=10000, queries_count=200):
    from benchmark.datasets.tgif_loader import load_caption_data
    tsv=Path(tsv) if tsv else ROOT/'data'/'tgif-v1.0.tsv'
    out=Path(out)
    if out.exists():
        raise FileExistsError(f'Output {out} already exists. Choose a new --output-dir, or delete it ONLY after backing up results/cache.')
    dataframe=load_caption_data(tsv)
    rows=[]
    by_gif=defaultdict(list)
    freq=Counter(norm(t) for t in dataframe['caption'])
    for i,(url,caption) in enumerate(dataframe[['url','caption']].itertuples(index=False,name=None)):
        gid='gif_'+hashlib.sha256(url.encode('utf-8')).hexdigest()[:20]
        d={'doc_id':f'doc_{i:07d}','gif_id':gid,'caption':caption,'url':url}
        rows.append(d)
        by_gif[gid].append(d)
    eligible=[]
    for gid,group in by_gif.items():
        if len(group)<2:continue
        choices=[d for d in group if freq[norm(d['caption'])]==1]
        if choices:eligible.append((gid,choices))
    rng=random.Random(seed)
    rng.shuffle(eligible)
    if len(eligible)<queries_count:
        raise ValueError(f'Only {len(eligible)} eligible GIFs; need {queries_count}')
    chosen=eligible[:queries_count]
    heldout=[rng.choice(choices) for _,choices in chosen]
    removed={d['doc_id'] for d in heldout}
    heldtexts={norm(d['caption']) for d in heldout}
    # Ensure every target has at least one caption in the selected corpus.
    mandatory=[]
    for gid,_ in chosen:
        candidate=next((d for d in by_gif[gid] if d['doc_id'] not in removed and norm(d['caption']) not in heldtexts),None)
        if candidate is None:raise ValueError('Cannot preserve target GIF '+gid)
        mandatory.append(candidate)
    if docs_count<len(mandatory):raise ValueError('docs_count too small')
    base={d['doc_id'] for d in mandatory}
    pool=[d for d in rows if d['doc_id'] not in removed and d['doc_id'] not in base and norm(d['caption']) not in heldtexts]
    if len(pool)<docs_count-len(mandatory):raise ValueError('Not enough captions to build corpus')
    selected_ids=base|{d['doc_id'] for d in rng.sample(pool,docs_count-len(mandatory))}
    corpus=[d for d in rows if d['doc_id'] in selected_ids]
    queries=[]
    for i,d in enumerate(heldout,1):
        queries.append({'query_id':f'A_{i:04d}','query':d['caption'],'relevant_ids':[d['gif_id']],
                        'source':'A','split':'test' if i%10<3 else 'validation',
                        'heldout_doc_id':d['doc_id'],'origin':'original_tgif_human_caption'})
    out.mkdir(parents=True)
    write_jsonl(out/'corpus.jsonl',corpus)
    write_jsonl(out/'queries.jsonl',queries)
    validate(out,expected_docs=docs_count,expected_queries=queries_count)
    metadata={'protocol':'TGIF source-A human-caption heldout','seed':seed,'captions':len(corpus),
              'queries':len(queries),'corpus_sha256':digest(out/'corpus.jsonl'),
              'queries_sha256':digest(out/'queries.jsonl'),'note':'Independent regenerated split; do not compare with results of another split.'}
    (out/'summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rebuild',action='store_true',help='Create a new dataset instead of validating existing')
    ap.add_argument('--output-dir',type=Path,default=DEFAULT)
    ap.add_argument('--tsv',type=Path)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.rebuild:build(args.tsv,args.output_dir,args.seed)
    else:validate(args.output_dir)

if __name__=='__main__':main()
