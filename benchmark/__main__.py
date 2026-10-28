"""One entry point for all GIFt benchmark operations: python -m benchmark --help."""
import importlib
import sys

COMMANDS={
    'prepare': ('benchmark.datasets.build_dataset1','main','Validate frozen dataset or rebuild a new 10k/200 split'),
    'embed': ('benchmark.retrieval.embed','main','Compute corpus/query embeddings; cached in data/benchmark/embeddings/'),
    'semantic1': ('benchmark.retrieval.semantic_dataset1','main','Dataset 1 semantic retrieval, saved vectors only'),
    'keyword1': ('benchmark.retrieval.keyword_dataset1','main','Dataset 1 BM25 keyword retrieval'),
    'hybrid1': ('benchmark.retrieval.hybrid_dataset1','main','Dataset 1 BM25 + semantic full-rank RRF'),
    'compare1': ('benchmark.reports.compare_dataset1','build_report','Dataset 1 comparison CSV+HTML, five methods'),
    'semantic2': ('benchmark.retrieval.semantic_dataset2','main','Dataset 2 semantic ranking and report'),
}

def main():
    if len(sys.argv)<2 or sys.argv[1] in ('-h','--help','help'):
        print('GIFt benchmark — commands (run from GIFt-main root):\n')
        for cmd, (_,_,desc) in COMMANDS.items():
            print(f'  python -m benchmark {cmd:<11} {desc}')
        print('\nUse python -m benchmark <command> --help to see options.')
        return
    cmd=sys.argv[1]
    if cmd not in COMMANDS:
        raise SystemExit(f'Unknown command: {cmd}. Run python -m benchmark --help')
    sys.argv=[sys.argv[0],*sys.argv[2:]]
    module,function,_=COMMANDS[cmd]
    getattr(importlib.import_module(module),function)()

if __name__=='__main__':main()
