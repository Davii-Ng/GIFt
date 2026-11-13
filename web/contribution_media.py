"""Bounded, cancellable FFmpeg processing for short personal videos."""
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_SOURCE_SECONDS = 15.0
MAX_GIF_SECONDS = 10.0
FPS = 12
VIDEO_SUFFIXES = {'.mp4', '.mov', '.webm', '.m4v'}
CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


class MediaError(ValueError):
    pass


class Cancelled(MediaError):
    pass


def ffmpeg():
    return os.getenv('FFMPEG_PATH') or imageio_ffmpeg.get_ffmpeg_exe()


def run_media(arguments, *, timeout=90, cancelled=lambda: False, progress=lambda value: None, duration=1):
    """Use argument arrays and a bounded process; user strings never become shell code."""
    with tempfile.TemporaryFile() as log, tempfile.TemporaryFile() as progress_log:
        process = subprocess.Popen(
            [ffmpeg(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '2',
             '-filter_threads', '1', '-filter_complex_threads', '1', '-protocol_whitelist', 'file,pipe',
             '-format_whitelist', 'mov,matroska,webm', '-progress', 'pipe:1', '-y', *arguments],
            stdin=subprocess.DEVNULL, stdout=progress_log, stderr=log, creationflags=CREATE_FLAGS,
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                if cancelled():
                    raise Cancelled('Creation cancelled.')
                if time.monotonic() - started > timeout:
                    raise MediaError('This video took too long to process. Try a shorter or smaller clip.')
                progress_log.seek(0)
                matches = re.findall(rb'out_time_us=(\d+)', progress_log.read())
                if matches:
                    progress(min(0.98, int(matches[-1]) / 1_000_000 / max(duration, 0.1)))
                time.sleep(0.1)
            if cancelled():
                raise Cancelled('Creation cancelled.')
            if process.returncode:
                raise MediaError('This video could not be decoded. Try an MP4 (H.264), MOV or WebM video.')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def probe_video(path: Path):
    """Read FFmpeg's decoded output metadata, including phone rotation."""
    try:
        result = subprocess.run([ffmpeg(), '-nostdin', '-hide_banner', '-threads', '2',
                                 '-protocol_whitelist', 'file,pipe', '-format_whitelist', 'mov,matroska,webm',
                                 '-i', str(path), '-map', '0:v:0', '-an', '-sn', '-dn', '-frames:v', '1',
                                 '-vf', 'scale=2:2', '-f', 'null', '-'],
                                capture_output=True, timeout=20, creationflags=CREATE_FLAGS)
        if result.returncode:
            raise MediaError('This file is not a readable video. Try MP4 (H.264), MOV or WebM.')
        header = result.stderr.decode('utf-8', errors='replace').split('Output #0')[0]
        duration_match = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', header)
        size_match = re.search(r'Video:[^\n]*?[, ](\d{2,5})x(\d{2,5})(?:[ ,\[])', header)
        if not duration_match or not size_match:
            raise MediaError('This file has no readable video duration or dimensions.')
        hours, minutes, seconds = map(float, duration_match.groups())
        duration = hours * 3600 + minutes * 60 + seconds
        width, height = map(int, size_match.groups())
        rotation = re.search(r'rotation of (-?\d+(?:\.\d+)?) degrees', header)
        if rotation and round(float(rotation.group(1))) % 180:
            width, height = height, width
        if not math.isfinite(duration) or not 0.1 <= duration <= MAX_SOURCE_SECONDS + 0.15:
            raise MediaError('Choose a video between 0.1 and 15 seconds long.')
        if min(width, height) < 16 or max(width, height) > 4096 or width * height > 9_000_000:
            raise MediaError('Choose a video up to 4K with a valid video track.')
        return {'width': width, 'height': height, 'duration': duration}
    except MediaError:
        raise
    except Exception:
        raise MediaError('This file is not a readable video. Try MP4 (H.264), MOV or WebM.') from None


def preview_video(source, destination, meta):
    run_media(['-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
               '-vf', "scale=w='min(854,iw)':h='min(480,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1",
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
               '-map_metadata', '-1', '-movflags', '+faststart', str(destination)], duration=meta['duration'])


def geometry(width, height, settings):
    rotation = settings.get('rotation', 0)
    if rotation in (90, 270):
        width, height = height, width
    ratio = {'original': width / height, 'square': 1, 'portrait': 9 / 16, 'landscape': 16 / 9}[settings['aspect']]
    crop_width = min(width, round(height * ratio))
    crop_height = min(height, round(width / ratio))
    x = round((width - crop_width) * settings.get('crop_x', 0.5))
    y = round((height - crop_height) * settings.get('crop_y', 0.5))
    factor = min(1, 480 / crop_height, 854 / crop_width)
    out_width = max(2, round(crop_width * factor / 2) * 2)
    out_height = max(2, round(crop_height * factor / 2) * 2)
    return crop_width, crop_height, x, y, out_width, out_height


def render_gif(source, destination, meta, settings, *, cancelled=lambda: False, progress=lambda value: None):
    cw, ch, x, y, width, height = geometry(meta['width'], meta['height'], settings)
    filters = []
    rotation = settings['rotation']
    if rotation == 90:
        filters.append('transpose=1')
    elif rotation == 180:
        filters.extend(['hflip', 'vflip'])
    elif rotation == 270:
        filters.append('transpose=2')
    filters.extend([f'crop={cw}:{ch}:{x}:{y}', f'fps={FPS}', f'scale={width}:{height}:flags=lanczos', 'setsar=1'])
    graph = ','.join(filters) + ',split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a'
    duration = settings['end'] - settings['start']
    run_media(['-ss', str(settings['start']), '-t', str(duration), '-i', str(source),
               '-an', '-sn', '-dn', '-filter_complex', graph, '-loop', '0', '-map_metadata', '-1', str(destination)],
              timeout=120, cancelled=cancelled, progress=progress, duration=duration)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise MediaError('The GIF could not be saved. Please try again.')
    if destination.stat().st_size > 30 * 1024 * 1024:
        destination.unlink(missing_ok=True)
        raise MediaError('The GIF is too large. Choose a shorter segment and try again.')
    return {'width': width, 'height': height, 'duration': duration, 'bytes': destination.stat().st_size}
