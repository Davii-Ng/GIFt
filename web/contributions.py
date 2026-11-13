"""Persistent per-Firebase-user GIF creation, private media, and public vector search."""
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, FiniteFloat, model_validator

from gift.search import get_model
from .contribution_search import (TEXT_LIMIT, VECTOR_VERSION, community_rank, lexical_priority,
                                  merge_results, metadata_vector, normalize_details, query_parts)
from .contribution_auth import bearer, current_contributor
from .contribution_media import (Cancelled, MediaError, MAX_GIF_SECONDS, MAX_SOURCE_SECONDS,
                                 MAX_UPLOAD_BYTES, VIDEO_SUFFIXES, preview_video, probe_video, render_gif)

logger = logging.getLogger('gifly.contributions')
router = APIRouter(prefix='/api/contributions', tags=['Your contribution'])
ROOT = Path(__file__).resolve().parent
TTL = 24 * 60 * 60


class GIFDetails(BaseModel):
    title: str = Field(default='', max_length=TEXT_LIMIT)
    description: str = Field(default='', max_length=TEXT_LIMIT)

    @model_validator(mode='after')
    def clean_details(self):
        self.title, self.description = normalize_details(self.title, self.description)
        return self


class CreateGIF(GIFDetails):
    upload_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    visibility: Literal['private', 'public'] = 'private'
    start: FiniteFloat = Field(default=0, ge=0, le=15)
    end: FiniteFloat = Field(gt=0, le=15.15)
    aspect: Literal['original', 'square', 'portrait', 'landscape'] = 'original'
    rotation: Literal[0, 90, 180, 270] = 0
    crop_x: FiniteFloat = Field(default=0.5, ge=0, le=1)
    crop_y: FiniteFloat = Field(default=0.5, ge=0, le=1)

    @model_validator(mode='after')
    def valid_segment(self):
        if not 0.1 <= self.end - self.start <= MAX_GIF_SECONDS + 0.001:
            raise ValueError('Select between 0.1 and 10 seconds for your GIF.')
        return self


class UpdateGIF(GIFDetails):
    visibility: Literal['private', 'public']


class VisibilityUpdate(BaseModel):
    visibility: Literal['private', 'public']

def row_dict(row):
    return dict(row) if row is not None else None


class ContributionService:
    def __init__(self, root=None, model_factory=get_model):
        self.root = Path(root or os.getenv('CONTRIBUTION_DATA_DIR') or ROOT / '.local' / 'contributions')
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ('uploads', 'files'):
            (self.root / folder).mkdir(exist_ok=True)
        self.db_path = self.root / 'contributions.sqlite3'
        self.model_factory = model_factory
        self.lock = threading.RLock()
        self.index_lock = threading.RLock()
        self.cancellations = {}
        self.upload_slots = threading.BoundedSemaphore(2)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='gif-render')
        self.collection = None
        self.chroma = None
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, suffix TEXT NOT NULL,
                    width INTEGER NOT NULL, height INTEGER NOT NULL, duration REAL NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS uploads_owner ON uploads(owner);
                CREATE TABLE IF NOT EXISTS contributions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, upload_id TEXT NOT NULL,
                    title TEXT NOT NULL, description TEXT NOT NULL, visibility TEXT NOT NULL,
                    status TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '', settings TEXT NOT NULL,
                    width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
                    duration REAL NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0,
                    vector TEXT, created REAL NOT NULL, updated REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS contributions_owner ON contributions(owner, created);
                CREATE INDEX IF NOT EXISTS contributions_public ON contributions(visibility, status);
            ''')
            db.execute("UPDATE contributions SET status='failed', error='Creation was interrupted. You can try again.', updated=? WHERE status IN ('queued','rendering','indexing')", (time.time(),))
            if 'vector_version' not in {row['name'] for row in db.execute('PRAGMA table_info(contributions)')}:
                db.execute('ALTER TABLE contributions ADD COLUMN vector_version INTEGER NOT NULL DEFAULT 0')
            db.executescript('''
                CREATE VIRTUAL TABLE IF NOT EXISTS contribution_keywords USING fts5(
                    gif_id UNINDEXED, title, description, tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TRIGGER IF NOT EXISTS contribution_keywords_insert AFTER INSERT ON contributions BEGIN
                    INSERT INTO contribution_keywords SELECT new.id, new.title, new.description
                    WHERE new.visibility='public' AND new.status='ready';
                END;
                CREATE TRIGGER IF NOT EXISTS contribution_keywords_update AFTER UPDATE ON contributions BEGIN
                    DELETE FROM contribution_keywords WHERE gif_id=old.id;
                    INSERT INTO contribution_keywords SELECT new.id, new.title, new.description
                    WHERE new.visibility='public' AND new.status='ready';
                END;
                CREATE TRIGGER IF NOT EXISTS contribution_keywords_delete AFTER DELETE ON contributions BEGIN
                    DELETE FROM contribution_keywords WHERE gif_id=old.id;
                END;
                INSERT INTO contribution_keywords
                    SELECT id,title,description FROM contributions
                    WHERE visibility='public' AND status='ready'
                    AND id NOT IN (SELECT gif_id FROM contribution_keywords);
            ''')
        self.cleanup()
        try:
            self.reindex_metadata()
        except Exception:
            logger.exception('Community metadata reindex pending; existing catalog search remains available')

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.db_path, timeout=20)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def close(self):
        with self.lock:
            for event in self.cancellations.values():
                event.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.chroma is not None:
            self.chroma.close()

    def file_path(self, gif_id):
        return self.root / 'files' / f'{gif_id}.gif'

    def source_path(self, upload):
        return self.root / 'uploads' / f"{upload['id']}{upload['suffix']}"

    def preview_path(self, upload_id):
        return self.root / 'uploads' / f'{upload_id}.preview.mp4'

    def cleanup(self):
        cutoff = time.time() - TTL
        with self.lock, self.db() as db:
            expired = db.execute("SELECT * FROM uploads WHERE created < ? AND id NOT IN (SELECT upload_id FROM contributions WHERE status IN ('queued','rendering','indexing'))", (cutoff,)).fetchall()
            for row in expired:
                self.source_path(row).unlink(missing_ok=True)
                self.preview_path(row['id']).unlink(missing_ok=True)
                db.execute('DELETE FROM uploads WHERE id=?', (row['id'],))
            # An interrupted upload never becomes a row; remove its orphaned temporary files.
            known = {row['id'] for row in db.execute('SELECT id FROM uploads')}
            for path in (self.root / 'uploads').iterdir():
                if path.is_file() and path.name.split('.')[0] not in known and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)

    def get(self, gif_id, owner=None):
        with self.db() as db:
            row = row_dict(db.execute('SELECT * FROM contributions WHERE id=?', (gif_id,)).fetchone())
        if row is None or owner is not None and row['owner'] != owner:
            raise HTTPException(404, 'GIF not found.')
        return row

    def upload(self, owner, file):
        self.cleanup()
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(415, 'Choose an MP4, MOV or WebM video.')
        with self.lock, self.db() as db:
            if db.execute('SELECT count(*) FROM uploads WHERE owner=?', (owner,)).fetchone()[0] >= 3:
                raise HTTPException(429, 'Finish or discard an existing upload before adding another video.')
        upload_id = uuid.uuid4().hex
        source = self.root / 'uploads' / f'{upload_id}{suffix}'
        preview = self.preview_path(upload_id)
        try:
            size = 0
            with source.open('xb') as target:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, 'Choose a video smaller than 50 MB.')
                    target.write(chunk)
            meta = probe_video(source)
            preview_video(source, preview, meta)
            with self.lock, self.db() as db:
                if db.execute('SELECT count(*) FROM uploads WHERE owner=?', (owner,)).fetchone()[0] >= 3:
                    raise HTTPException(429, 'You already have three active uploads.')
                db.execute('INSERT INTO uploads VALUES (?,?,?,?,?,?,?)',
                           (upload_id, owner, suffix, meta['width'], meta['height'], meta['duration'], time.time()))
            return {'id': upload_id, **meta, 'bytes': size,
                    'preview_url': f'/api/contributions/uploads/{upload_id}/preview'}
        except Exception as error:
            source.unlink(missing_ok=True)
            preview.unlink(missing_ok=True)
            if isinstance(error, MediaError):
                raise HTTPException(422, str(error)) from None
            raise

    def get_upload(self, upload_id, owner):
        with self.db() as db:
            upload = row_dict(db.execute('SELECT * FROM uploads WHERE id=? AND owner=?', (upload_id, owner)).fetchone())
        if upload and upload['created'] < time.time() - TTL:
            self.cleanup()
            upload = None
        if not upload or not self.source_path(upload).is_file():
            raise HTTPException(404, 'This video has expired. Please upload it again.')
        return upload

    def remove_upload(self, upload_id, owner):
        upload = self.get_upload(upload_id, owner)
        with self.lock, self.db() as db:
            if db.execute("SELECT 1 FROM contributions WHERE upload_id=? AND status IN ('queued','rendering','indexing')", (upload_id,)).fetchone():
                raise HTTPException(409, 'Cancel GIF creation before discarding this video.')
            db.execute('DELETE FROM uploads WHERE id=?', (upload_id,))
        self.source_path(upload).unlink(missing_ok=True)
        self.preview_path(upload_id).unlink(missing_ok=True)

    def create(self, owner, body):
        upload = self.get_upload(body.upload_id, owner)
        if body.end > upload['duration'] + 0.1:
            raise HTTPException(422, 'The selected segment extends past the end of the video.')
        gif_id = f'contrib_{uuid.uuid4().hex}'
        now = time.time()
        with self.lock, self.db() as db:
            if db.execute("SELECT count(*) FROM contributions WHERE owner=? AND status IN ('queued','rendering','indexing')", (owner,)).fetchone()[0] >= 2:
                raise HTTPException(429, 'You can create two GIFs at a time. Please wait for one to finish.')
            if db.execute("SELECT count(*) FROM contributions WHERE status IN ('queued','rendering','indexing')").fetchone()[0] >= 12:
                raise HTTPException(429, 'The creation queue is full. Please try again shortly.')
            if db.execute("SELECT count(*) FROM contributions WHERE owner=? AND status!='cancelled'", (owner,)).fetchone()[0] >= 200:
                raise HTTPException(429, 'Your library is full. Delete a GIF before creating another.')
            if db.execute('SELECT 1 FROM contributions WHERE upload_id=?', (body.upload_id,)).fetchone():
                raise HTTPException(409, 'A GIF has already been created from this upload. Check Your contribution.')
            db.execute('''INSERT INTO contributions
                (id,owner,upload_id,title,description,visibility,status,settings,created,updated)
                VALUES (?,?,?,?,?,?,'queued',?,?,?)''',
                (gif_id, owner, body.upload_id, body.title, body.description, body.visibility,
                 json.dumps(body.model_dump()), now, now))
            self.cancellations[gif_id] = threading.Event()
            self.executor.submit(self.process, gif_id)
        return self.serialize(self.get(gif_id, owner), owner=True)

    def update_state(self, gif_id, **values):
        values['updated'] = time.time()
        with self.db() as db:
            db.execute('UPDATE contributions SET ' + ','.join(f'{key}=?' for key in values) + ' WHERE id=?', (*values.values(), gif_id))

    def index(self):
        if self.collection is None:
            import chromadb
            from chromadb.config import Settings
            from gift.config import MODEL_NAME
            self.chroma = chromadb.PersistentClient(path=str(self.root / 'chroma'), settings=Settings(anonymized_telemetry=False))
            self.collection = self.chroma.get_or_create_collection(
                'gifly_public_contributions_v1', embedding_function=None,
                configuration={'hnsw': {'space': 'cosine'}}, metadata={'model': MODEL_NAME})
        return self.collection

    def publish_vector(self, row, vector):
        if row['visibility'] == 'public':
            document = row['title'] if row['title'] == row['description'] else row['title'] + '\n' + row['description']
            self.index().upsert(ids=[row['id']], embeddings=[vector], documents=[document],
                                metadatas=[{'source': 'Community', 'category': 'general'}])
        elif self.collection is not None or (self.root / 'chroma').exists():
            self.index().delete(ids=[row['id']])

    def reindex_metadata(self):
        """Upgrade existing descriptions-only vectors without changing user metadata or media."""
        with self.db() as db:
            rows = db.execute("SELECT * FROM contributions WHERE status='ready' AND (vector_version<? OR vector IS NULL)", (VECTOR_VERSION,)).fetchall()
        if not rows:
            return 0
        with self.index_lock:
            for row in rows:
                vector = metadata_vector(self.model_factory(), row['title'], row['description'])
                self.publish_vector(row, vector)
                with self.db() as db:
                    db.execute('UPDATE contributions SET vector=?,vector_version=? WHERE id=?',
                               (json.dumps(vector), VECTOR_VERSION, row['id']))
        logger.info('Updated name/description vectors for %s contributions', len(rows))
        return len(rows)

    def process(self, gif_id):
        event = self.cancellations[gif_id]
        output = self.file_path(gif_id)
        last_progress = -1

        def progress(value):
            nonlocal last_progress
            percent = min(85, 5 + int(value * 80))
            if percent != last_progress:
                last_progress = percent
                self.update_state(gif_id, progress=percent)

        try:
            row = self.get(gif_id)
            if event.is_set():
                raise Cancelled('Creation cancelled.')
            upload = self.get_upload(row['upload_id'], row['owner'])
            self.update_state(gif_id, status='rendering', progress=5, error='')
            meta = render_gif(self.source_path(upload), output, upload, json.loads(row['settings']),
                              cancelled=event.is_set, progress=progress)
            self.update_state(gif_id, status='indexing', progress=90)
            vector = metadata_vector(self.model_factory(), row['title'], row['description'])
            if event.is_set():
                raise Cancelled('Creation cancelled.')
            with self.index_lock:
                self.publish_vector(row, vector)
                self.update_state(gif_id, status='ready', progress=100, vector=json.dumps(vector), vector_version=VECTOR_VERSION, **meta)
            self.source_path(upload).unlink(missing_ok=True)
            self.preview_path(upload['id']).unlink(missing_ok=True)
            with self.db() as db:
                db.execute('DELETE FROM uploads WHERE id=?', (upload['id'],))
        except Cancelled:
            output.unlink(missing_ok=True)
            self.update_state(gif_id, status='cancelled', progress=0, error='Creation cancelled.')
        except Exception as error:
            logger.warning('GIF creation failed (%s): %s', gif_id, type(error).__name__)
            output.unlink(missing_ok=True)
            message = str(error) if isinstance(error, MediaError) else 'Could not finish creating and indexing this GIF. Please try again.'
            self.update_state(gif_id, status='failed', error=message)
        finally:
            with self.lock:
                self.cancellations.pop(gif_id, None)

    def serialize(self, row, *, owner=False):
        item = {key: row[key] for key in ('id','title','description','visibility','status','progress','error','width','height','duration','bytes','created','updated')}
        item.update(source='Community', category='general', ratio=row['width'] / row['height'] if row['height'] else 1.2,
                    url=f"/api/contributions/{row['id']}/file" if row['status'] == 'ready' else None)
        if owner:
            with self.db() as db:
                item['can_retry'] = row['status'] in ('failed','cancelled') and bool(db.execute('SELECT 1 FROM uploads WHERE id=?', (row['upload_id'],)).fetchone())
        return item

    def own(self, owner):
        with self.db() as db:
            rows = db.execute('SELECT * FROM contributions WHERE owner=? ORDER BY created DESC', (owner,)).fetchall()
        return [self.serialize(row, owner=True) for row in rows]

    def public(self):
        with self.db() as db:
            rows = db.execute("SELECT * FROM contributions WHERE visibility='public' AND status='ready' ORDER BY created DESC").fetchall()
        return [self.serialize(row) for row in rows]

    def public_by_ids(self, ids):
        if not ids:
            return []
        with self.db() as db:
            rows = db.execute("SELECT * FROM contributions WHERE visibility='public' AND status='ready' AND id IN (" + ','.join('?' for _ in ids) + ')', ids).fetchall()
        by_id = {row['id']: self.serialize(row) for row in rows}
        return [by_id[key] for key in ids if key in by_id]

    def feedback_vectors(self, ids):
        if not ids:
            return []
        with self.db() as db:
            rows = db.execute("SELECT id,vector FROM contributions WHERE visibility='public' AND status='ready' AND id IN (" + ','.join('?' for _ in ids) + ')', ids).fetchall()
        return [(row['id'], json.loads(row['vector'])) for row in rows if row['vector']]

    def search(self, query, limit=24, vector=None, hidden=(), category='all'):
        if category not in ('all', 'general'):
            return []
        with self.db() as db:
            count = db.execute("SELECT count(*) FROM contributions WHERE visibility='public' AND status='ready'").fetchone()[0]
        if not count:
            return []
        with self.index_lock:
            collection = self.index()
            if vector is None:
                vector = self.model_factory().encode(query, normalize_embeddings=True).tolist()
            # Rocchio returns a NumPy float32 array. list(array) retains NumPy
            # scalars, which Chroma rejects inside nested Python lists.
            query_vector = [float(value) for value in vector]
            hits = (collection.query(query_embeddings=[query_vector], n_results=min(collection.count(), limit + len(hidden)), include=['distances'])
                    if collection.count() else {'ids': [[]], 'distances': [[]]})
            items = {row['id']: row for row in self.public_by_ids(hits['ids'][0])}
            scored = {}
            words, phrase = query_parts(query)
            for key, distance in zip(hits['ids'][0], hits['distances'][0]):
                if key not in items or key in hidden:
                    continue
                priority = lexical_priority(query, items[key]['title'], items[key]['description'])
                if phrase and not priority:
                    continue
                score = round(max(-1.0, min(1.0, 1 - distance)), 4)
                scored[key] = dict(items[key], score=score, similarity=score, match_priority=priority)
            # Keyword hits are retrieved independently of nearest-neighbor cutoffs.
            # This also finds names in a different language from the embedding model.
            if words:
                expression = ('"' + ' '.join(words) + '"' if phrase else ' AND '.join('"' + word + '"' for word in dict.fromkeys(words)))
                exclusion = ' AND c.id NOT IN (' + ','.join('?' for _ in hidden) + ')' if hidden else ''
                with self.db() as db:
                    matches = db.execute('''SELECT c.* FROM contribution_keywords
                        JOIN contributions c ON c.id=contribution_keywords.gif_id
                        WHERE contribution_keywords MATCH ? AND c.visibility='public' AND c.status='ready' '''
                        + exclusion + ' ORDER BY bm25(contribution_keywords,0,3,1),c.created DESC LIMIT ?',
                        (expression, *hidden, limit)).fetchall()
                q = np.asarray(query_vector, dtype=np.float32)
                for row in matches:
                    priority = lexical_priority(query, row['title'], row['description'])
                    if not priority:
                        continue
                    if row['id'] in scored:
                        scored[row['id']]['match_priority'] = priority
                        continue
                    stored = np.asarray(json.loads(row['vector']), dtype=np.float32)
                    score = float(np.dot(stored, q) / max(float(np.linalg.norm(stored) * np.linalg.norm(q)), 1e-12))
                    score = round(max(-1.0, min(1.0, score)), 4)
                    scored[row['id']] = dict(self.serialize(row), score=score, similarity=score, match_priority=priority)
            return sorted(scored.values(), key=community_rank, reverse=True)[:limit]

    def edit(self, gif_id, owner, body):
        with self.index_lock:
            row = self.get(gif_id, owner)
            if row['status'] != 'ready':
                raise HTTPException(409, 'Wait for GIF creation to finish before editing its details.')
            vector = (json.loads(row['vector']) if body.title == row['title'] and body.description == row['description'] and row['vector'] and row['vector_version'] == VECTOR_VERSION else
                      metadata_vector(self.model_factory(), body.title, body.description))
            values = dict(title=body.title, description=body.description, visibility=body.visibility,
                          vector=json.dumps(vector), vector_version=VECTOR_VERSION)
            if body.visibility == 'private':
                # Revoke access even when removal from a stale index is temporarily unavailable.
                self.update_state(gif_id, **values)
                try:
                    self.publish_vector(dict(row, **values), vector)
                except Exception:
                    logger.warning('Deferred index removal for private GIF %s', gif_id)
            else:
                try:
                    self.publish_vector(dict(row, **values), vector)
                except Exception:
                    raise HTTPException(503, 'Could not update public search. Your previous settings are unchanged. Please try again.') from None
                self.update_state(gif_id, **values)
            return self.serialize(self.get(gif_id, owner), owner=True)

    def cancel(self, gif_id, owner):
        with self.lock:
            row = self.get(gif_id, owner)
            if row['status'] not in ('queued','rendering','indexing'):
                return self.serialize(row, owner=True)
            event = self.cancellations.get(gif_id)
            if event:
                event.set()
            return self.serialize(self.get(gif_id, owner), owner=True)

    def retry(self, gif_id, owner):
        with self.lock:
            row = self.get(gif_id, owner)
            if row['status'] not in ('failed','cancelled') or gif_id in self.cancellations:
                raise HTTPException(409, 'This GIF cannot be retried right now.')
            self.get_upload(row['upload_id'], owner)
            with self.db() as db:
                if db.execute("SELECT count(*) FROM contributions WHERE owner=? AND status IN ('queued','rendering','indexing')", (owner,)).fetchone()[0] >= 2:
                    raise HTTPException(429, 'You can create two GIFs at a time.')
                if db.execute("SELECT count(*) FROM contributions WHERE status IN ('queued','rendering','indexing')").fetchone()[0] >= 12:
                    raise HTTPException(429, 'The queue is full. Please try again shortly.')
            self.update_state(gif_id, status='queued', progress=0, error='')
            self.cancellations[gif_id] = threading.Event()
            self.executor.submit(self.process, gif_id)
        return self.serialize(self.get(gif_id, owner), owner=True)

    def delete(self, gif_id, owner):
        with self.lock, self.index_lock:
            row = self.get(gif_id, owner)
            if gif_id in self.cancellations:
                self.cancellations[gif_id].set()
                raise HTTPException(409, 'Creation is being cancelled. Delete the GIF once it has stopped.')
            # Remove the authoritative record first; public reads never trust index metadata.
            with self.db() as db:
                db.execute('DELETE FROM contributions WHERE id=? AND owner=?', (gif_id, owner))
                upload = row_dict(db.execute('SELECT * FROM uploads WHERE id=?', (row['upload_id'],)).fetchone())
                db.execute('DELETE FROM uploads WHERE id=?', (row['upload_id'],))
            self.file_path(gif_id).unlink(missing_ok=True)
            if upload:
                self.source_path(upload).unlink(missing_ok=True)
                self.preview_path(upload['id']).unlink(missing_ok=True)
            if self.collection is not None or (self.root / 'chroma').exists():
                self.index().delete(ids=[gif_id])


def service(request: Request):
    return request.app.state.contributions


@router.get('/status')
def capabilities():
    return {'max_upload_bytes': MAX_UPLOAD_BYTES, 'max_video_seconds': MAX_SOURCE_SECONDS,
            'max_gif_seconds': MAX_GIF_SECONDS, 'height': 480, 'fps': 12}


@router.post('/uploads', status_code=201)
def upload_video(file: UploadFile = File(), user=Depends(current_contributor), store=Depends(service)):
    acquired = store.upload_slots.acquire(blocking=False)
    if not acquired:
        file.file.close()
        raise HTTPException(429, 'Two videos are being prepared. Please try again shortly.')
    try:
        return store.upload(user['uid'], file)
    finally:
        file.file.close()
        store.upload_slots.release()


@router.get('/uploads/{upload_id}/preview')
def upload_preview(upload_id: str, user=Depends(current_contributor), store=Depends(service)):
    store.get_upload(upload_id, user['uid'])
    path = store.preview_path(upload_id)
    if not path.is_file():
        raise HTTPException(404, 'Video preview not found.')
    return FileResponse(path, media_type='video/mp4', headers={'Cache-Control': 'private, no-store'})


@router.delete('/uploads/{upload_id}', status_code=204)
def discard_upload(upload_id: str, user=Depends(current_contributor), store=Depends(service)):
    store.remove_upload(upload_id, user['uid'])


@router.post('', status_code=202)
def create_gif(body: CreateGIF, user=Depends(current_contributor), store=Depends(service)):
    return store.create(user['uid'], body)


@router.get('')
def own_contributions(user=Depends(current_contributor), store=Depends(service)):
    return {'gifs': store.own(user['uid'])}


@router.get('/{gif_id}/file')
def gif_file(gif_id: str, download: bool = False, credentials=Depends(bearer), store=Depends(service)):
    row = store.get(gif_id)
    if row['status'] != 'ready':
        raise HTTPException(404, 'GIF not found.')
    if row['visibility'] != 'public':
        try:
            user = current_contributor(credentials)
        except HTTPException:
            raise HTTPException(404, 'GIF not found.') from None
        if row['owner'] != user['uid']:
            raise HTTPException(404, 'GIF not found.')
    path = store.file_path(gif_id)
    if not path.is_file():
        raise HTTPException(404, 'GIF not found.')
    return FileResponse(path, media_type='image/gif', filename=f'{gif_id}.gif' if download else None,
                        headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})


@router.get('/{gif_id}')
def contribution_detail(gif_id: str, user=Depends(current_contributor), store=Depends(service)):
    return store.serialize(store.get(gif_id, user['uid']), owner=True)


@router.put('/{gif_id}')
def edit_contribution(gif_id: str, body: UpdateGIF, user=Depends(current_contributor), store=Depends(service)):
    return store.edit(gif_id, user['uid'], body)


@router.patch('/{gif_id}/visibility')
def change_visibility(gif_id: str, body: VisibilityUpdate, user=Depends(current_contributor), store=Depends(service)):
    with store.index_lock:
        row = store.get(gif_id, user['uid'])
        return store.edit(gif_id, user['uid'], UpdateGIF(title=row['title'], description=row['description'],
                                                       visibility=body.visibility))


@router.post('/{gif_id}/cancel')
def cancel_contribution(gif_id: str, user=Depends(current_contributor), store=Depends(service)):
    return store.cancel(gif_id, user['uid'])


@router.post('/{gif_id}/retry', status_code=202)
def retry_contribution(gif_id: str, user=Depends(current_contributor), store=Depends(service)):
    return store.retry(gif_id, user['uid'])


@router.delete('/{gif_id}', status_code=204)
def delete_contribution(gif_id: str, user=Depends(current_contributor), store=Depends(service)):
    store.delete(gif_id, user['uid'])
