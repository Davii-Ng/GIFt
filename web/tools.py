"""Browser access to relevance feedback, video search and benchmark reports."""
import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, FiniteFloat

from gift.config import DATA_DIR, ROOT
from gift.gemini.feedback import adjust_query, forget_similar, hidden_for, normalize
from gift.video.search import embed_text, load_index, rank_rows

from .auto_search import POOL_SIZE
from .contributions import merge_results

router = APIRouter(prefix="/api/tools")
logger = logging.getLogger("gifly.tools")
VIDEO_INDEX = Path(DATA_DIR) / "processed" / "gemini_embedding_2"
VIDEO_DIR = Path(DATA_DIR) / "videos"
REPORTS = Path(ROOT) / "benchmark" / "results"
VIDEO_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
               ".m4v": "video/mp4", ".mkv": "video/x-matroska"}
REPORT_TYPES = {".html", ".csv", ".json"}
benchmark_lock = Lock()


def gemini_status():
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"):
        return "Set GEMINI_API_KEY (or configure Vertex AI) in the server environment."
    if importlib.util.find_spec("google") is None or importlib.util.find_spec("google.genai") is None:
        return "Install the Gemini SDK with python -m pip install google-genai."
    return ""


def gif_status():
    if not (Path(DATA_DIR) / "chroma_db_gemini" / "chroma.sqlite3").is_file():
        return "Build the GIF index with python -m gift.gemini.index."
    return gemini_status()


def video_status():
    if not all((VIDEO_INDEX / name).is_file() for name in ("video_vectors.npy", "video_metadata.jsonl")):
        return "Add clips to data/videos/inbox and run python -m gift.video.embed."
    return gemini_status()


@router.get("/status")
def status(request: Request):
    local = "" if request.app.state.retriever else request.app.state.message
    return {"local": local, "gemini": gif_status(), "video": video_status()}


class Dislike(BaseModel):
    gif_id: str = Field(max_length=100)
    query: str = Field(max_length=300)
    query_vector: list[FiniteFloat] = Field(min_length=1, max_length=768)


class FeedbackSearch(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    model: Literal["local", "gemini", "auto"] = "local"
    limit: int = Field(default=12, ge=1, le=60)
    # Supplying an offset enables relevance-ordered, positive-only search pages.
    # Existing recommendation clients can omit it to retain their current behavior.
    offset: int | None = Field(default=None, ge=0)
    category: Literal["all", "reactions", "animals", "celebration", "mood", "general"] = "all"
    liked: list[str] = Field(default_factory=list, max_length=100)
    disliked: list[str] = Field(default_factory=list, max_length=100)
    memory: list[Dislike] = Field(default_factory=list, max_length=100)
    unhide: bool = False


@lru_cache(maxsize=64)
def local_vector(model, query):
    return tuple(float(x) for x in model.encode([query], normalize_embeddings=True)[0])


@lru_cache(maxsize=64)
def gemini_vector(query):
    from gift.gemini.search import embed_query
    return tuple(embed_query(query))


@router.post("/feedback")
def feedback_search(body: FeedbackSearch, request: Request):
    query = body.query.strip()
    if not query:
        raise HTTPException(422, "Enter a search description.")
    if body.model == "gemini":
        if message := gif_status():
            raise HTTPException(503, message)
    elif not request.app.state.retriever:
        raise HTTPException(503, request.app.state.message)
    try:
        if body.model == "gemini":
            from gift.gemini.search import _load
            _, collection = _load()
            vector = gemini_vector(query)
        else:
            retriever = request.app.state.retriever
            collection = retriever.collection
            vector = local_vector(retriever.model, query)

        memory = [item.model_dump() for item in body.memory if len(item.query_vector) == len(vector)]
        if body.unhide:
            memory = forget_similar(memory, vector)
        hidden = hidden_for(memory, vector) - set(body.disliked) - set(body.liked)
        votes = list(dict.fromkeys(body.liked + body.disliked))
        adjusted = vector
        if votes:
            found = collection.get(ids=votes, include=["embeddings"])
            vote_ids = list(found['ids'])
            vote_vectors = [list(value) for value in found['embeddings']] if len(vote_ids) else []
            if body.model != 'gemini':
                try:
                    community_votes = request.app.state.contributions.feedback_vectors(votes)
                except Exception:
                    logger.exception('Community feedback vectors unavailable; retaining catalog feedback')
                    community_votes = []
                for key, values in community_votes:
                    if len(values) == len(vector):
                        vote_ids.append(key); vote_vectors.append(values)
            if vote_ids:
                index = {"vectors": normalize(vote_vectors),
                         "row_of": {key: i for i, key in enumerate(vote_ids)}}
                adjusted = adjust_query(vector, index, body.liked, body.disliked)

        auto = request.app.state.auto_search if body.model == "auto" and not votes else None
        method = auto.choose(query) if auto else "semantic"
        paged = body.offset is not None
        wanted = (body.offset or 0) + body.limit + int(paged)
        pool = wanted if method == "semantic" else max(wanted, POOL_SIZE)
        total = collection.count()
        count = min(total, pool + len(hidden))
        items = []
        while count:
            items = []
            hits = collection.query(query_embeddings=[[float(value) for value in adjusted]], n_results=count,
                                    where={"category": body.category} if body.model != "gemini" and body.category != "all" else None,
                                    include=["documents", "metadatas", "distances"])
            for gif_id, description, meta, distance in zip(
                hits["ids"][0], hits["documents"][0], hits["metadatas"][0], hits["distances"][0]
            ):
                url = meta.get("url", "")
                if gif_id not in hidden and urlsplit(url).scheme in {"http", "https"}:
                    item = {**meta, "id": gif_id, "url": url, "description": description, "score": 1 - distance}
                    if body.model == "gemini":
                        item.update(title=description[:75], source="Gemini",
                                    favorite_id=request.app.state.id_by_url.get(url))
                    items.append(item)
            # Hidden or unusable sources must not create an artificial last page.
            # Chroma query has no offset, so expand the prefix only as needed.
            positive = sum(item['score'] > 0 for item in items)
            if (not paged or positive >= pool or count >= total or len(hits['ids'][0]) < count
                    or not hits['distances'][0] or hits['distances'][0][-1] >= 1):
                break
            count = min(total, max(count * 2, count + body.limit))
        if method != "semantic":
            items = auto.rank(query, method, {item["id"]: item for item in items}, body.category, hidden)
            # Keyword-only candidates also need a cosine score for a fair merge.
            missing = [item['id'] for item in items if 'score' not in item]
            if missing:
                data = collection.get(ids=missing, include=['embeddings'])
                direction = normalize(adjusted)
                scores = {key: float(normalize(values) @ direction)
                          for key, values in zip(data['ids'], data['embeddings'])}
                items = [dict(item, score=scores[item['id']], similarity=scores[item['id']])
                         if item['id'] in scores else item for item in items]
        community = []
        if body.model != 'gemini':
            try:
                community = request.app.state.contributions.search(query, wanted, vector=adjusted,
                                                                  hidden=hidden, category=body.category)
            except Exception:
                logger.exception('Community search unavailable; retaining catalog results')
                community = []
        if paged or community:
            items = merge_results(items, community, wanted)
        response = {"hidden": len(hidden), "vector": vector, "memory": memory if body.unhide else None}
        if paged:
            # Sort even when the optional community index is empty/unavailable.
            # Keyword candidates improve recall but cannot bypass the zero cutoff.
            matches = [item for item in items if item.get('score', -1) > 0]
            page = matches[body.offset:body.offset + body.limit]
            response.update(results=page, has_more=len(matches) > body.offset + body.limit,
                            next_offset=body.offset + len(page))
        else:
            response['results'] = items[:body.limit]
        return response
    except Exception:
        logger.exception('Feedback search failed')
        raise HTTPException(503, 'Search is temporarily unavailable. Please try again.') from None


@lru_cache(maxsize=1)
def video_index(vectors_mtime, metadata_mtime):
    return load_index(VIDEO_INDEX)


def current_video_index():
    vectors = (VIDEO_INDEX / "video_vectors.npy").stat().st_mtime_ns
    metadata = (VIDEO_INDEX / "video_metadata.jsonl").stat().st_mtime_ns
    return video_index(vectors, metadata)


def video_file(item):
    path = (VIDEO_DIR / item.get("relative_path", "")).resolve()
    if path.is_relative_to(VIDEO_DIR.resolve()) and path.suffix.lower() in VIDEO_TYPES and path.is_file():
        return path
    return None


@router.get("/videos")
def video_search(q: str = Query(min_length=1, max_length=300), limit: int = Query(6, ge=1, le=20)):
    if not q.strip():
        raise HTTPException(422, "Enter a video description.")
    if message := video_status():
        raise HTTPException(503, message)
    try:
        vectors, metadata = current_video_index()
        if not metadata:
            raise HTTPException(503, "The video index is empty. Add clips and rebuild it.")
        return {"results": [
            {"name": Path(metadata[row]["relative_path"]).name, "score": score,
             "url": f"/api/tools/videos/{row}/file" if video_file(metadata[row]) else None}
            for row, score in rank_rows(vectors, embed_text(q.strip()), limit)
        ]}
    except HTTPException:
        raise
    except Exception as error:
        logger.warning("Video search failed: %s", type(error).__name__)
        raise HTTPException(503, "Video search failed. Check the Gemini settings and the video index.") from None


@router.get("/videos/{row}/file")
def stream_video(row: int):
    try:
        _, metadata = current_video_index()
        path = video_file(metadata[row]) if 0 <= row < len(metadata) else None
    except (OSError, ValueError):
        path = None
    if path is None:
        raise HTTPException(404, "Video file not found.")
    return FileResponse(path, media_type=VIDEO_TYPES[path.suffix.lower()])


@router.get("/reports")
def reports():
    return {"reports": [
        path.relative_to(REPORTS).as_posix()
        for path in sorted(REPORTS.glob("dataset*/**/*"))
        if path.is_file() and path.suffix.lower() in REPORT_TYPES
        and path.resolve().is_relative_to(REPORTS.resolve())
    ]}


@router.get("/reports/{name:path}")
def report_file(name: str):
    path = (REPORTS / name).resolve()
    if (not path.is_relative_to(REPORTS.resolve()) or path.suffix.lower() not in REPORT_TYPES
            or not path.is_file()):
        raise HTTPException(404, "Benchmark report not found.")
    if path.suffix.lower() == ".html":
        return FileResponse(path, media_type="text/html", headers={
            "Content-Security-Policy": "sandbox allow-scripts; default-src 'none'; "
                                       "img-src https: data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            "X-Content-Type-Options": "nosniff",
        })
    return FileResponse(path, filename=path.name)


@router.post("/benchmarks/{method}")
def run_benchmark(method: Literal["bm25", "semantic", "hybrid"],
                  model: Literal["minilm", "qwen3", "bge-m3"] = "minilm"):
    from benchmark.reports.compare_dataset1 import build_report
    from benchmark.retrieval import keyword_dataset1, keyword_dataset2

    if not benchmark_lock.acquire(blocking=False):
        raise HTTPException(409, "A benchmark is already in progress.")
    try:
        if method == "bm25":
            keyword_dataset1.run()
            keyword_dataset2.run()
        else:
            from benchmark.retrieval import embed, semantic_dataset1, semantic_dataset2

            embed.main(["--model", model, "--full"])
            semantic_dataset1.main(["--model", model])
            semantic_dataset2.main(["--model", model])
            if method == "hybrid":
                from benchmark.retrieval import hybrid_dataset1, hybrid_dataset2

                hybrid_dataset1.run(model=model)
                hybrid_dataset2.run(model=model)
            keyword_dataset2.build_comparison()
        build_report()
        return {"status": "complete"}
    except Exception:
        logger.exception("Benchmark failed")
        raise HTTPException(500, "Could not complete the benchmark. Check the model cache and server log.") from None
    finally:
        benchmark_lock.release()
