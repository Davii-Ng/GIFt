# Create GIF and Your contribution

Run `start-web.ps1` from the project root after installing `requirements.txt`. The bundled `imageio-ffmpeg` dependency supplies FFmpeg; a separate system-wide FFmpeg installation is not needed. `FFMPEG_PATH` can override its executable.

The working project retains Firebase Authentication and favorites. Upstream `b5fb4ab2e8db04f98c71fe156d7106009c302c3d` has been integrated, including `gift/`, GIF Mirror in `apps/streamlit_app.py`, and video indexing. Old top-level Python commands forward to the new modules.

## Workflow

1. Sign in with the existing Firebase account.
2. Open **Create GIF**. Choose or drop an MP4, MOV, M4V or WebM video, at most 15 seconds and 50 MB. The server validates and decodes the file, then makes an H.264 preview for browser compatibility.
3. Select a 0.1–10 second segment. Keep the original ratio or crop to 1:1, 9:16 or 16:9; rotate in 90-degree steps and position the crop by dragging or sliders.
4. Add a name, a description, or both. At least one must contain text. If only one is provided, its entire trimmed text becomes both name and description (up to 1,000 characters each). Choose **Just me** (default) or **Everyone**. The same rule applies when editing a completed GIF.
5. Create the GIF. The task continues while browsing or after reloading; open **Your contribution** to check it. The original is deleted after a successful render. Failed/cancelled tasks can be retried while their source remains available.
6. Download the completed GIF, edit its description and visibility, copy its public media link, or delete it. Click the Public/Private badge on a ready contribution to switch visibility directly; the badge changes after the server confirms the update.

GIFs loop forever, have no audio, use 12 fps and a fixed output bound of 854 × 480 pixels. Aspect ratio is retained and small sources are not enlarged. Portrait GIFs are typically 270 × 480; square GIFs are 480 × 480. There is no text tool or quality picker. Estimated size is advisory; the finished file reports the actual size.

## Ownership, database and search

- The server verifies Firebase ID tokens against Google's signing certificates, including audience, issuer, expiry and UID. `FIREBASE_PROJECT_ID` may be set explicitly; otherwise it is read from the public `web/firebase-config.js`. No service-account private key is required for token verification.
- Canonical job, description, visibility and ownership records are in `web/.local/contributions/contributions.sqlite3`, keyed by Firebase UID. GIF files are under `files/` and temporary videos under `uploads/`. `CONTRIBUTION_DATA_DIR` overrides this root.
- The browser mirrors contribution metadata to `users/{uid}/contributions/{gifId}` in Firestore using the signed-in user's permissions. **Publish the updated project-root `firestore.rules` to enable this new collection.** Favorites rules remain intact. If cloud sync fails, the server library remains durable, the failure is retained internally and logged to the browser console; no success is claimed for the failed sync.
- Community embeddings use the existing `gift.search.get_model()` MiniLM instance: a normalized blend of 60% name and 40% description embeddings. A single embedding is used when both fields are identical. Existing uploaded GIFs are automatically upgraded to this representation without changing their names, descriptions, media files or timestamps.
- A separate SQLite FTS5 index retrieves public GIFs matching names or descriptions, including quoted phrases, independently of semantic nearest-neighbor cutoffs. Candidates are merged using a single relevance weight: catalog GIFs ×1.0, public community GIFs ×1.02. There are no additional name/description keyword bonuses. Zero and negative similarities receive no boost. For example, a community similarity of 0.59 ranks as 0.6018, ahead of a catalog result at 0.60; a community similarity of 0.58 ranks as 0.5916 and stays below it even with an exact name match. Community GIFs have no guaranteed top position. Reported cosine similarity remains unchanged. Like/dislike adjustments and hidden-result filters still apply.
- In Explore, Search and Saved, open a GIF to access Download in its preview; cards have no small download overlay. Your contribution retains its library download action. Catalog downloads use canonical GIF IDs and a bounded server-side fetch, so they do not depend on the remote site's browser CORS settings. Unavailable or invalid original sources return an error logged to the browser console. Contribution downloads retain the same owner/public access checks as media previews.
- Errors, warnings and system confirmations are console-only, prefixed `[gifly:context]`. Toasts, error banners, account/edit/delete errors, raw render failures and cloud-sync warnings are not displayed. The logger prints messages rather than request/account objects, ignores aborted requests and deduplicates recurring render/media/sync failures. Input validation still blocks invalid submissions; native validation bubbles are suppressed. Normal progress, job state, required input hints and the explicit end-of-search message remain part of the interface.
- Website GIF searches use `POST /api/tools/feedback` with `offset` and a page `limit` (up to 60 per request). Pages retain the query, category, votes and hidden-result memory and merge catalog/community candidates by relevance with the community ×1.02 weight. Only similarity scores greater than zero are included; there is no 60-result total cap. The response includes `next_offset` and `has_more`. Load more appends results without clearing the query, and a final click displays an end-of-results message below the list. Feedback/category/query changes start again at the first page. Clients omitting `offset` keep the existing non-paginated behavior.
- Only public, completed GIFs go into the separate Chroma and keyword collections. Private, unpublished and deleted GIFs are excluded at query time as well. They appear in the homepage, `/api/search`, and the website's feedback search. The main TGIF index and its fingerprint are unchanged.
- Publish the current `firestore.rules` again if using cloud metadata sync: both persisted fields are nonempty, with a 1,000-character maximum, including names copied from long descriptions.
- Public media requests are checked against the canonical database on every request. Private media requires the owner's bearer token; browsers retrieve it as an authenticated blob. Changing to private or deleting immediately prevents anonymous access and future search results, even if a stale vector exists. Previously downloaded copies cannot be recalled.
- The browser clears private previews and pending operations when accounts change. Firebase sync checks the expected UID before writing.

Back up the **entire** contribution directory (SQLite, GIFs and Chroma), along with the Firebase project configuration and rules. Deploy this version as **one API process** with a persistent disk: its bounded two-worker render queue is local to that process. Multiple API workers or ephemeral deployment storage require a shared durable task queue and object storage before use.

## Limits and recovery

- Up to two conversions at once, twelve pending conversions globally, and two pending jobs per user.
- Up to three temporary uploads and 200 contributions per account.
- A conversion times out after 120 seconds. Files larger than 30 MB after rendering require a shorter selection.
- Unused source uploads expire after 24 hours; cleanup runs on startup, each upload and every five minutes. Source access also checks expiry.
- Interrupted jobs become retryable failures after restart. A successfully deleted original cannot be used to re-trim a completed GIF; upload the source again for a different edit.
- Render commands use argument lists, restricted input formats/protocols, generated file IDs, and no user-provided shell commands. Metadata and media files are not mounted as static directories.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/contributions/status` | Public limits and output defaults |
| `POST /api/contributions/uploads` | Owner-authenticated multipart upload and preview preparation |
| `GET /api/contributions/uploads/{id}/preview` | Owner-only normalized MP4 preview |
| `DELETE /api/contributions/uploads/{id}` | Discard an unused source |
| `POST /api/contributions` | Submit a GIF job |
| `GET /api/contributions` | Current user's library |
| `GET /api/contributions/{id}` | Owner-only job status/details |
| `PUT /api/contributions/{id}` | Edit title, description and visibility |
| `PATCH /api/contributions/{id}/visibility` | Owner-only visibility switch without replacing title or description |
| `POST /api/contributions/{id}/cancel` | Cancel queued/running creation |
| `POST /api/contributions/{id}/retry` | Retry an interrupted/failed job with retained source |
| `GET /api/contributions/{id}/file` | Public GIF, or authenticated owner's private GIF |
| `DELETE /api/contributions/{id}` | Delete the owner's GIF, source and index entry |
| `GET /api/gifs/{id}/download` | Download a known catalog GIF or an accessible contribution |

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s web/tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --test web/tests/api-client.test.mjs web/tests/firebase-account.test.mjs web/tests/contributions.test.mjs web/tests/gif-actions.test.mjs web/tests/system-log.test.mjs
```

The contribution tests generate real video, decode real GIF output, and exercise ownership, public search, unpublishing, description re-indexing, cancellation, retries and persistence. Token verification is mocked in unit tests; the integration smoke test also verified a real temporary Firebase account and the real MiniLM search, then removed its account and media.
