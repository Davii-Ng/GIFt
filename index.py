"""Compatibility entry point; implementation lives in gift.index."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("gift.index", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("gift.index")
