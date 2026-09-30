"""Settings shared by every script. Change things here, not in the other files."""
import os

# Folder layout. Everything generated (download + database) lives in data/, which git ignores.
ROOT = os.path.dirname(os.path.abspath(__file__))  # this repo's folder
DATA_DIR = os.path.join(ROOT, "data")
TSV_PATH = os.path.join(DATA_DIR, "tgif-v1.0.tsv")  # downloaded dataset
DB_PATH = os.path.join(DATA_DIR, "chroma_db")  # vector database

# Dataset
TSV_URL = "https://raw.githubusercontent.com/raingo/TGIF-Release/master/data/tgif-v1.0.tsv"

# Model and database
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast model: 384 numbers per sentence
COLLECTION = "gif-search"  # a "collection" is like a table in the database
BATCH_SIZE = 64  # encode this many descriptions at a time (faster than one by one)
