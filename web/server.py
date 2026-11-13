"""Phase 2: serve the website and query the local Chroma index."""
import asyncio
import logging
from random import Random
from contextlib import asynccontextmanager, suppress
from time import perf_counter
from typing import Literal


from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .engine import MODEL_NAME, ROOT, Retriever, catalog_fingerprint, close_database, read_catalog

from .auto_search import AutoSearch, calibrate
from .tools import benchmark_lock, router as tools_router
from .contributions import ContributionService, merge_results, router as contributions_router
from .contribution_media import MAX_UPLOAD_BYTES
from .gif_downloads import router as downloads_router

logger = logging.getLogger("gifly")


def calibrate_search(auto, fingerprint):
    try:
        with benchmark_lock:
            auto.methods = calibrate(fingerprint, auto)
        logger.info("Automatic search policy ready: %s", auto.methods)
    except Exception:
        logger.exception("Automatic benchmark failed. Semantic search remains available.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.contributions = ContributionService()
    async def clean_uploads():
        while True:
            await asyncio.sleep(300)
            await asyncio.to_thread(app.state.contributions.cleanup)
    upload_cleanup = asyncio.create_task(clean_uploads())
    app.state.auto_search = None
    calibration = None
    app.state.catalog = []
    app.state.by_id = {}
    app.state.id_by_url = {}
    app.state.source = "GIF"
    app.state.categories = []
    app.state.caption_count = 0
    app.state.retriever = None
    app.state.message = "Search is not ready. Run python -m web.ingest, then restart the server."
    try:
        records = read_catalog()
        app.state.catalog = records
        app.state.by_id = {row["id"]: row for row in records}
        app.state.id_by_url = {row["url"]: row["id"] for row in records}
        sources = {row.get("source", "Sample") for row in records}
        app.state.source = next(iter(sources)) if len(sources) == 1 else "GIF"
        app.state.categories = sorted({row["category"] for row in records} - {"general"})
        app.state.caption_count = sum(row.get("caption_count", 1) for row in records)
        retriever = Retriever()
        fingerprint = catalog_fingerprint(records)
        if (retriever.collection.metadata or {}).get("catalog_fingerprint") != fingerprint:
            raise ValueError("Catalog changed. Run python -m web.ingest before restarting.")
        app.state.retriever = retriever
    except Exception:
        close_database()
        logger.exception("Search unavailable. Run python -m web.ingest, then restart the server.")
    if app.state.retriever is not None:
        try:
            app.state.auto_search = AutoSearch(records)
            if app.state.source == "TGIF" and len(records) >= 1000:
                calibration = asyncio.create_task(asyncio.to_thread(calibrate_search, app.state.auto_search, fingerprint))
        except Exception:
            logger.exception("Keyword index unavailable. Semantic search remains available.")
    try:
        yield
    finally:
        upload_cleanup.cancel()
        with suppress(asyncio.CancelledError):
            await upload_cleanup
        await asyncio.to_thread(app.state.contributions.close)
        if calibration is not None:
            await calibration
        if app.state.retriever is not None:
            app.state.retriever.close()



app = FastAPI(title="gifly", lifespan=lifespan)
app.include_router(tools_router)
app.include_router(contributions_router)
app.include_router(downloads_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://{host}:{port}" for host in ("127.0.0.1", "localhost", "[::1]") for port in (5500, 5501)],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Accept", "Content-Type", "Authorization"],
)
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")


class UploadLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') != '/api/contributions/uploads':
            return await self.app(scope, receive, send)
        from starlette.responses import JSONResponse
        if scope.get('method') == 'POST' and not dict(scope.get('headers', [])).get(b'authorization', b'').lower().startswith(b'bearer '):
            return await JSONResponse({'detail': 'Log in to upload a video.'}, status_code=401)(scope, receive, send)
        limit = MAX_UPLOAD_BYTES + 1024 * 1024
        headers = dict(scope.get('headers', []))
        try:
            if int(headers.get(b'content-length', b'0')) > limit:
                return await JSONResponse({'detail': 'Choose a video smaller than 50 MB.'}, status_code=413)(scope, receive, send)
        except ValueError:
            return await JSONResponse({'detail': 'Invalid upload size.'}, status_code=400)(scope, receive, send)
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get('body', b''))
            if size > limit:
                raise HTTPException(413, 'Choose a video smaller than 50 MB.')
            return message
        await self.app(scope, bounded_receive, send)


app.add_middleware(UploadLimitMiddleware)


@app.middleware("http")
async def protect_account_requests(request: Request, call_next):
    # JSON mutations are accepted only from this site or the local preview.
    if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
        origin = request.headers.get("origin")
        allowed = {str(request.base_url).rstrip("/")} | {
            f"http://{host}:{port}" for host in ("127.0.0.1", "localhost", "[::1]") for port in (5500, 5501)
        }
        if origin and origin not in allowed:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "This request origin is not allowed."}, status_code=403)
    response = await call_next(request)
    if request.url.path.startswith(("/api/auth", "/api/favorites", "/api/contributions")):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
def health():
    ready = app.state.retriever is not None
    return {
        "ready": ready, "backend": "chroma", "model": MODEL_NAME,
        "count": len(app.state.catalog), "source": app.state.source,
        "message": "Ready to search." if ready else app.state.message,
        "caption_count": app.state.caption_count,
    }


@app.get("/api/catalog")
def catalog(limit: int = Query(default=24, ge=1, le=60), offset: int = Query(default=0, ge=0),
            category: Literal["all", "reactions", "animals", "celebration", "mood", "general"] = "all",
            ids: list[str] | None = Query(default=None, max_length=60),
            seed: int | None = Query(default=None, ge=0, le=2**32 - 1)):
    if ids is not None:
        extra = {row['id']: row for row in app.state.contributions.public_by_ids(ids)}
        records = [app.state.by_id.get(key) or extra[key] for key in dict.fromkeys(ids) if key in app.state.by_id or key in extra]
        return {"gifs": records, "total": len(records), "has_more": False, "source": app.state.source}
    records = [*app.state.catalog, *app.state.contributions.public()]
    if category != "all":
        records = [row for row in records if row["category"] == category]
    if seed is not None:
        records = records.copy()
        Random(seed).shuffle(records)
    return {"gifs": records[offset:offset + limit], "total": len(records), "offset": offset,
            "has_more": offset + limit < len(records), "source": app.state.source,
            "categories": app.state.categories}


@app.get("/api/search")
def search(q: str = Query(min_length=1, max_length=300), limit: int = Query(default=24, ge=1, le=60),
           category: Literal["all", "reactions", "animals", "celebration", "mood", "general"] = "all"):
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Enter a keyword or a feeling to search for.")
    start = perf_counter()
    try:
        results = app.state.retriever.search(query, limit, category) if app.state.retriever else []
        extra = app.state.contributions.search(query, limit, category=category)
        if app.state.retriever is None and not extra:
            raise HTTPException(status_code=503, detail=app.state.message)
        results = merge_results(results, extra, limit) if extra else results
    except HTTPException:
        raise
    except Exception:
        logger.exception("Search failed")
        raise HTTPException(status_code=503, detail="Search is unavailable right now. Please try again.") from None
    return {"query": query, "results": results, "count": len(results), "elapsed_ms": round((perf_counter() - start) * 1000)}


@app.get("/")
@app.get("/index.html", include_in_schema=False)
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/style.css")
def stylesheet():
    return FileResponse(ROOT / "style.css", media_type="text/css")


@app.get("/app.js")
def javascript():
    return FileResponse(ROOT / "app.js", media_type="text/javascript")


@app.get("/api-client.js")
def api_javascript():
    return FileResponse(ROOT / "api-client.js", media_type="text/javascript")


@app.get("/tools.js", include_in_schema=False)
def tools_javascript():
    return FileResponse(ROOT / "tools.js", media_type="text/javascript")


@app.get("/firebase-account.js", include_in_schema=False)
def firebase_account_script():
    return FileResponse(ROOT / "firebase-account.js", media_type="text/javascript")


@app.get("/firebase-config.js", include_in_schema=False)
def firebase_config_script():
    return FileResponse(ROOT / "firebase-config.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})


@app.get('/contributions.js', include_in_schema=False)
def contributions_script():
    return FileResponse(ROOT / 'contributions.js', media_type='text/javascript')


@app.get('/contributions.css', include_in_schema=False)
def contributions_styles():
    return FileResponse(ROOT / 'contributions.css', media_type='text/css')


@app.get('/gif-actions.js', include_in_schema=False)
def gif_actions_script():
    return FileResponse(ROOT / 'gif-actions.js', media_type='text/javascript')


@app.get('/system-log.js', include_in_schema=False)
def system_log_script():
    return FileResponse(ROOT / 'system-log.js', media_type='text/javascript')
