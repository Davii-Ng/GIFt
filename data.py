"""TGIF caption data loader.

`load_data` is retained for the original one-result-per-GIF Chroma demo.
`load_caption_data` preserves every distinct GIF/caption pair for the benchmark.
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


def load_data(limit=None):
    """Backward-compatible loader for the original Chroma/Streamlit demo.

    Returns columns `url` and `description` with one caption per unique GIF.
    The benchmark MUST use `load_caption_data`, not this function.
    """
    df = load_caption_data().drop_duplicates(subset="url")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be a positive integer")
        df = df.sample(n=min(limit, len(df)), random_state=42)
    return df.rename(columns={"caption": "description"}).reset_index(drop=True)
