"""Download known catalog GIFs without depending on remote browser CORS headers."""
import ipaddress
import re
import socket
import tempfile
import time
from urllib.parse import quote, urljoin, urlsplit

import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from .contribution_auth import bearer
from .contributions import gif_file, service

router = APIRouter(prefix='/api/gifs', tags=['GIF downloads'])
MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024
GIF_HEADERS = (b'GIF87a', b'GIF89a')


def validate_remote_url(url):
    try:
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise HTTPException(502, 'The original GIF source is unavailable.')
        try:
            literal = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            literal = None
        if literal and (not literal.is_global or literal.is_multicast):
            raise HTTPException(502, 'The original GIF source is unavailable.')
        addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global or ipaddress.ip_address(item[4][0]).is_multicast for item in addresses):
            raise HTTPException(502, 'The original GIF source is unavailable.')
    except (socket.gaierror, ValueError):
        raise HTTPException(502, 'The original GIF source is unavailable.') from None


def remote_gif(url):
    destination = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode='w+b')
    started = time.monotonic()
    try:
        for redirect in range(4):
            validate_remote_url(url)
            with requests.get(url, stream=True, allow_redirects=False, timeout=(5, 10), headers={
                'Accept': 'image/gif', 'User-Agent': 'Mozilla/5.0 (compatible; gifly GIF download)',
            }) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get('Location')
                    if not location or redirect == 3:
                        raise HTTPException(502, 'The original GIF source is unavailable.')
                    url = urljoin(url, location)
                    continue
                if response.status_code != 200:
                    raise HTTPException(502, 'This GIF source is no longer available. Try opening the original GIF.')
                try:
                    length = int(response.headers.get('Content-Length', 0))
                except ValueError:
                    length = 0
                if length > MAX_DOWNLOAD_BYTES:
                    raise HTTPException(413, 'This GIF is larger than the 30 MB download limit.')
                size = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if time.monotonic() - started > 30:
                        raise HTTPException(504, 'The GIF source took too long to respond. Please try again.')
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise HTTPException(413, 'This GIF is larger than the 30 MB download limit.')
                    destination.write(chunk)
                destination.seek(0)
                if destination.read(6) not in GIF_HEADERS:
                    raise HTTPException(502, 'The original source did not return a valid GIF.')
                destination.seek(0)
                return destination, size
        raise HTTPException(502, 'The original GIF source is unavailable.')
    except requests.Timeout:
        destination.close()
        raise HTTPException(504, 'The GIF source took too long to respond. Please try again.') from None
    except requests.RequestException:
        destination.close()
        raise HTTPException(502, 'Cannot download this GIF from its original source right now.') from None
    except Exception:
        destination.close()
        raise


def chunks(file):
    try:
        while chunk := file.read(64 * 1024):
            yield chunk
    finally:
        file.close()


@router.get('/{gif_id}/download')
def download_gif(gif_id: str, request: Request, credentials=Depends(bearer), store=Depends(service)):
    if gif_id.startswith('contrib_'):
        return gif_file(gif_id, download=True, credentials=credentials, store=store)
    row = request.app.state.by_id.get(gif_id)
    if row is None:
        raise HTTPException(404, 'GIF not found in the current catalog.')
    # The client supplies an ID, never a remote URL. Only canonical catalog URLs are fetched.
    file, size = remote_gif(row['url'])
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', row.get('title') or gif_id).strip(' .')[:100] or 'gif'
    return StreamingResponse(chunks(file), media_type='image/gif', background=BackgroundTask(file.close), headers={
        'Content-Disposition': f"attachment; filename=\"gif.gif\"; filename*=UTF-8''{quote(name + '.gif', safe='')}",
        'Content-Length': str(size), 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
    })
