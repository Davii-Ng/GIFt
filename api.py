"""HTTP API for searching the local GIF index.

Run with: uvicorn api:app --host 127.0.0.1 --port 8000
"""
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from search import search_gif

MAX_QUERY_CHARS = 500
MAX_RESULTS = 20

app = FastAPI(title="GIF Search API", version="1.0.0")


@app.get("/health")
def health():
    """Report that the API process is running."""
    return {"status": "ok"}


@app.get("/search")
def search(
    q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_CHARS)],
    k: Annotated[int, Query(ge=1, le=MAX_RESULTS)] = 9,
):
    """Return GIFs matching a text query."""
    if not q.strip():
        raise HTTPException(status_code=422, detail="Query must not be blank.")

    try:
        return search_gif(q, k)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error