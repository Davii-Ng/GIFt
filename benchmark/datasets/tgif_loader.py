"""TGIF caption data loader.

Benchmark-only TGIF TSV loader; preserves every distinct GIF/caption pair.
This module does not modify your root-level data.py or original Chroma demo.
"""
from pathlib import Path

import pandas as pd
import requests

from config import DATA_DIR, TSV_PATH, TSV_URL


def ensure_tgif_tsv(tsv_path=TSV_PATH):
    """Download TGIF metadata once, leaving existing local data untouched."""
    path = Path(tsv_path)
    if path.exists():
        return path
    # Only download automatically for the project's default data location.
    if path.resolve() != Path(TSV_PATH).resolve():
        raise FileNotFoundError(f"Provided TSV does not exist: {path}")
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Downloading TGIF caption metadata to {path} ...")
    response = requests.get(TSV_URL, timeout=120)
    response.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_bytes(response.content)
    tmp.replace(path)
    return path


def load_caption_data(tsv_path=TSV_PATH):
    """Return (url, caption) rows with all DISTINCT captions retained per GIF.

    Cleaning: trim whitespace, remove missing URL/caption and exact
    (url, caption) duplicates. Do not remove duplicate URLs.
    """
    path = ensure_tgif_tsv(tsv_path)
    df = pd.read_csv(
        path, sep="\t", header=None, names=["url", "caption"],
        dtype=str, keep_default_na=False, on_bad_lines="error",
    )
    for column in ("url", "caption"):
        df[column] = df[column].str.strip()
    df = df.loc[(df["url"] != "") & (df["caption"] != "")]
    return df.drop_duplicates(subset=["url", "caption"]).reset_index(drop=True)

