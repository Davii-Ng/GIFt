"""Step 1: get the data.

TGIF is a public dataset of ~125,000 GIFs. Each row has two things:
  - url:         where the GIF lives on the internet
  - description: a sentence a human wrote about the GIF
We only download the text file (~18 MB). The GIF images stay on their original servers.
"""
import os

import pandas as pd
import requests

from config import DATA_DIR, TSV_PATH, TSV_URL


def load_data(limit=None):
    """Return a table (pandas DataFrame) with columns `url` and `description`.

    limit: keep only this many random rows. None = keep everything.
    """
    # Download only if we do not already have the file.
    if not os.path.exists(TSV_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        print("Downloading TGIF dataset...")
        response = requests.get(TSV_URL, timeout=60)
        response.raise_for_status()  # stop with an error if the download failed
        with open(TSV_PATH, "wb") as f:
            f.write(response.content)

    # The file is tab-separated and has no header row, so we name the columns ourselves.
    df = pd.read_csv(TSV_PATH, sep="\t", names=["url", "description"])

    # Clean up: drop empty rows and repeated GIFs.
    df = df.dropna().drop_duplicates(subset="url").reset_index(drop=True)

    if limit:
        # random_state=42 makes the "random" pick the same every run.
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
    return df
