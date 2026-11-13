import { requestJSON } from './api-client.js';
import { downloadFilename } from './gif-actions.js?v=gif-actions-relevance';
import { logSystem } from './system-log.js';

const MAX_BYTES = 50 * 1024 * 1024;
const ACTIVE = new Set(['queued', 'rendering', 'indexing']);
const sizeLabel = bytes => bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;

export function normalizeGIFDetails(title = '', description = '') {
  title = title.trim(); description = description.trim();
  if (!title && !description) throw new Error('Enter a name or a description. You can leave one field empty, but not both.');
  if ([title, description].some(value => [...value].length > 1000)) throw new Error('Use up to 1000 characters for each field.');
  return { title: title || description, description: description || title };
}

export function validateVideoFile(file) {
  if (!file || !/\.(mp4|mov|webm|m4v)$/i.test(file.name)) return 'Choose an MP4, MOV or WebM video.';
  if (file.size > MAX_BYTES) return 'Choose a video smaller than 50 MB.';
  if (!file.size) return 'This file is empty. Choose a video.';
  return '';
}

export function cropGeometry(width, height, aspect, rotation = 0, x = 0.5, y = 0.5) {
  if (rotation === 90 || rotation === 270) [width, height] = [height, width];
  const ratio = { original: width / height, square: 1, portrait: 9 / 16, landscape: 16 / 9 }[aspect];
  const cropWidth = Math.min(width, Math.round(height * ratio));
  const cropHeight = Math.min(height, Math.round(width / ratio));
  const factor = Math.min(1, 480 / cropHeight, 854 / cropWidth);
  return { width, height, cropWidth, cropHeight, x: Math.round((width - cropWidth) * x), y: Math.round((height - cropHeight) * y),
    outputWidth: Math.max(2, Math.round(cropWidth * factor / 2) * 2), outputHeight: Math.max(2, Math.round(cropHeight * factor / 2) * 2) };
}

export function initContributions({ base = '', getToken, syncCloud, requestLogin, toast, onNavigate }) {
  const $ = selector => document.querySelector(selector);
  const s = { user: null, revision: 0, page: 'explore', upload: null, uploadEpoch: 0, uploading: false,
    pendingFile: null, afterLogin: null, previewURL: null, frame: null, aspect: 'original', rotation: 0,
    start: 0, end: 0, job: null, items: [], filter: 'all', poll: null, loading: false, blobs: new Map(),
    cloud: new Map(), cloudError: false, editing: null, deleting: null, submitting: false, visibilityRequests: new Set() };
  const video = $('#creator-video');
  const canvas = $('#creator-canvas');
  const context = canvas.getContext('2d');
  const rotated = document.createElement('canvas');
  const rotatedContext = rotated.getContext('2d');

  function notice(message, target = '#creator-notice') {
    logSystem(message, { level: 'error', context: 'contributions' });
    $(target).textContent = ''; $(target).hidden = true;
  }

  async function tokenFor(revision = s.revision) {
    const token = await getToken(s.user?.id);
    if (revision !== s.revision || !s.user) throw new DOMException('Account changed.', 'AbortError');
    return token;
  }

  async function api(path = '', options = {}) {
    const revision = s.revision;
    const token = await tokenFor(revision);
    const result = await requestJSON(`/api/contributions${path}`, { base, ...options, headers: { Authorization: `Bearer ${token}` } });
    if (revision !== s.revision) throw new DOMException('Account changed.', 'AbortError');
    return result;
  }

  async function media(url, revision = s.revision) {
    const token = await tokenFor(revision);
    const response = await fetch(base + url, { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' });
    if (!response.ok) throw new Error('This preview is unavailable. Please refresh and try again.');
    const blob = await response.blob();
    if (revision !== s.revision) throw new DOMException('Account changed.', 'AbortError');
    return blob;
  }

  function show(page) {
    s.page = page;
    $('#explore-content').hidden = page !== 'explore';
    $('#creator-view').hidden = page !== 'creator';
    $('#contributions-view').hidden = page !== 'library';
    for (const [id, selected] of [['create-nav', page === 'creator'], ['contributions-nav', page === 'library']]) {
      $(`#${id}`).classList.toggle('active', selected); $(`#${id}`).setAttribute('aria-pressed', selected);
    }
    if (page !== 'explore') {
      for (const id of ['browse-nav', 'saved-nav']) { $(`#${id}`).classList.remove('active'); $(`#${id}`).setAttribute('aria-pressed', false); }
      video.pause(); onNavigate?.();
    }
    if (page === 'library') {
      if (!s.user) { s.afterLogin = 'library'; requestLogin(); renderLibrary(); }
      else loadLibrary();
    }
    if (page === 'creator' && s.previewURL) draw();
  }

  function revokeMedia() {
    if (s.previewURL) URL.revokeObjectURL(s.previewURL);
    s.previewURL = null;
    for (const url of s.blobs.values()) URL.revokeObjectURL(url);
    s.blobs.clear();
  }

  function settings() {
    return { start: s.start, end: s.end, aspect: s.aspect, rotation: s.rotation,
      crop_x: Number($('#crop-x').value), crop_y: Number($('#crop-y').value) };
  }

  function draw() {
    if (!video.videoWidth || !s.upload || $('#creator-canvas').hidden) return;
    const g = cropGeometry(video.videoWidth, video.videoHeight, s.aspect, s.rotation,
      Number($('#crop-x').value), Number($('#crop-y').value));
    if (rotated.width !== g.width || rotated.height !== g.height) { rotated.width = g.width; rotated.height = g.height; }
    if (canvas.width !== g.outputWidth || canvas.height !== g.outputHeight) { canvas.width = g.outputWidth; canvas.height = g.outputHeight; }
    rotatedContext.save(); rotatedContext.clearRect(0, 0, rotated.width, rotated.height);
    rotatedContext.translate(g.width / 2, g.height / 2); rotatedContext.rotate(s.rotation * Math.PI / 180);
    rotatedContext.drawImage(video, -video.videoWidth / 2, -video.videoHeight / 2); rotatedContext.restore();
    context.drawImage(rotated, g.x, g.y, g.cropWidth, g.cropHeight, 0, 0, canvas.width, canvas.height);
    $('#creator-time').textContent = `${video.currentTime.toFixed(1)} / ${s.upload.duration.toFixed(1)} s`;
  }

  function tick() {
    if (!video.paused && s.upload) {
      if (video.currentTime >= s.end - 0.03 || video.currentTime < s.start) video.currentTime = s.start;
      draw(); s.frame = requestAnimationFrame(tick);
    }
  }
  video.addEventListener('seeked', draw);
  video.addEventListener('loadeddata', draw);
  video.addEventListener('play', () => { cancelAnimationFrame(s.frame); tick(); $('#creator-play').innerHTML = 'Ⅱ <span>Pause preview</span>'; });
  video.addEventListener('pause', () => { cancelAnimationFrame(s.frame); $('#creator-play').innerHTML = '▶ <span>Play preview</span>'; });
  video.addEventListener('ended', () => { if (s.upload && s.page === 'creator' && !s.job) { video.currentTime = s.start; video.play().catch(() => {}); } });
  $('#creator-play').addEventListener('click', () => {
    if (video.paused) { if (video.currentTime >= s.end || video.currentTime < s.start) video.currentTime = s.start; video.play().catch(() => notice('Use your browser play controls to preview this clip.')); }
    else video.pause();
  });

  async function filmstrip(url, epoch) {
    const strip = $('#creator-filmstrip'); const ctx = strip.getContext('2d');
    const sampler = document.createElement('video'); sampler.muted = true; sampler.playsInline = true; sampler.src = url;
    const waitFor = name => new Promise((resolve, reject) => {
      const timer = setTimeout(() => { sampler.removeEventListener(name, done); reject(new Error('Preview timed out')); }, 5000);
      function done() { clearTimeout(timer); resolve(); }
      sampler.addEventListener(name, done, { once: true });
    });
    try {
      const ready = waitFor('loadeddata'); sampler.load(); await ready;
      for (let i = 0; i < 10; i++) {
        if (epoch !== s.uploadEpoch) return;
        const at = Math.min(s.upload.duration - 0.03, Math.max(0.01, (i + 0.5) / 10 * s.upload.duration));
        const sought = waitFor('seeked'); sampler.currentTime = at; await sought;
        const sw = sampler.videoWidth, sh = sampler.videoHeight;
        const cropWidth = Math.min(sw, sh * (strip.width / 10 / strip.height));
        ctx.drawImage(sampler, (sw - cropWidth) / 2, 0, cropWidth, sh, i * strip.width / 10, 0, strip.width / 10, strip.height);
      }
    } catch { /* Timeline controls still work when thumbnail extraction is unavailable. */ }
    finally { sampler.removeAttribute('src'); sampler.load(); }
  }

  function updateControls() {
    const valid = Boolean(s.upload && !s.uploading && !s.job && !s.submitting);
    $('#creator-edit-controls').disabled = !valid;
    $('#creator-play').disabled = !valid;
    $('#replace-video').disabled = s.uploading || Boolean(s.job);
    $('#create-submit').disabled = !valid || !($('#creator-gif-title').value.trim() || $('#creator-description').value.trim());
    const duration = Math.max(0, s.end - s.start);
    $('#segment-duration').textContent = `${duration.toFixed(1)} / 10 s`;
    $('#create-duration').textContent = s.upload ? `${duration.toFixed(1)} seconds` : '—';
    $('#create-size').textContent = s.upload ? `~${Math.max(0.2, duration * 0.65).toFixed(1)} MB` : '—';
    $('#description-count').textContent = `${$('#creator-description').value.length} / 1000`;
    for (const [id, value] of [['trim-start', s.start], ['start-seconds', s.start], ['trim-end', s.end], ['end-seconds', s.end]]) $(`#${id}`).value = value.toFixed(1);
    if (s.upload) {
      $('#timeline-selection').style.left = `${s.start / s.upload.duration * 100}%`;
      $('#timeline-selection').style.width = `${duration / s.upload.duration * 100}%`;
    }
    $('#crop-position').hidden = s.aspect === 'original';
    $('#creator-canvas').classList.toggle('draggable', s.aspect !== 'original');
    for (let i = 1; i <= 3; i++) $(`#creator-step-${i}`).classList.toggle('active', i === (s.job?.status === 'ready' ? 3 : s.upload ? 2 : 1));
  }

  function trim(which, value) {
    if (!s.upload || !Number.isFinite(value)) return;
    const max = Math.floor(s.upload.duration * 10) / 10;
    if (which === 'start') {
      s.start = Math.max(0, Math.min(max - 0.1, value));
      if (s.end <= s.start) s.end = Math.min(max, s.start + 0.1);
      if (s.end - s.start > 10) s.end = s.start + 10;
    } else {
      s.end = Math.max(0.1, Math.min(max, value));
      if (s.end <= s.start) s.start = Math.max(0, s.end - 0.1);
      if (s.end - s.start > 10) s.start = s.end - 10;
    }
    video.pause(); video.currentTime = which === 'start' ? s.start : Math.max(s.start, s.end - 0.05);
    updateControls(); draw();
  }
  for (const [id, which] of [['trim-start', 'start'], ['start-seconds', 'start'], ['trim-end', 'end'], ['end-seconds', 'end']]) {
    $(`#${id}`).addEventListener('input', event => trim(which, Number(event.target.value)));
  }
  document.querySelectorAll('[data-aspect]').forEach(button => button.addEventListener('click', () => {
    s.aspect = button.dataset.aspect;
    for (const item of document.querySelectorAll('[data-aspect]')) { const selected = item === button; item.classList.toggle('selected', selected); item.setAttribute('aria-pressed', selected); }
    $('#crop-x').value = $('#crop-y').value = '0.5'; updateControls(); draw();
  }));
  $('#rotate-video').addEventListener('click', () => { s.rotation = (s.rotation + 90) % 360; updateControls(); draw(); });
  for (const id of ['crop-x', 'crop-y']) $(`#${id}`).addEventListener('input', draw);
  let dragging = null;
  canvas.addEventListener('pointerdown', event => {
    if (!s.upload || s.job || s.aspect === 'original') return;
    dragging = { x: event.clientX, y: event.clientY, cropX: Number($('#crop-x').value), cropY: Number($('#crop-y').value) };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!dragging) return;
    const rect = canvas.getBoundingClientRect();
    $('#crop-x').value = Math.max(0, Math.min(1, dragging.cropX - (event.clientX - dragging.x) / rect.width));
    $('#crop-y').value = Math.max(0, Math.min(1, dragging.cropY - (event.clientY - dragging.y) / rect.height)); draw();
  });
  for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(event, () => dragging = null);

  async function discard(upload = s.upload) {
    if (!upload || !s.user) return;
    try { await api(`/uploads/${upload.id}`, { method: 'DELETE' }); } catch { /* Expired and consumed uploads are already removed. */ }
  }

  function uploadRequest(file, token, epoch) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest(); xhr.open('POST', base + '/api/contributions/uploads');
      xhr.setRequestHeader('Authorization', `Bearer ${token}`); xhr.responseType = 'json'; xhr.timeout = 180000;
      xhr.upload.addEventListener('progress', event => {
        if (epoch !== s.uploadEpoch) return;
        if (event.lengthComputable) $('#upload-meter').value = Math.round(event.loaded / event.total * 100);
        $('#upload-status').textContent = event.loaded === event.total ? 'Preparing your preview…' : 'Uploading your clip…';
      });
      xhr.onload = () => xhr.status >= 200 && xhr.status < 300 ? resolve(xhr.response) : reject(new Error(typeof xhr.response?.detail === 'string' ? xhr.response.detail : 'The upload failed. Please try again.'));
      xhr.onerror = () => reject(new Error('Upload interrupted. Check your connection and try again.'));
      xhr.ontimeout = () => reject(new Error('The upload timed out. Try a smaller video.'));
      const body = new FormData(); body.append('file', file); xhr.send(body);
    });
  }

  async function chooseFile(file) {
    const error = validateVideoFile(file); if (error) return notice(error);
    if (!s.user) { s.pendingFile = file; s.afterLogin = 'upload'; requestLogin(); return; }
    if (s.job && ACTIVE.has(s.job.status)) return notice('Finish or cancel your current GIF before choosing a new video.');
    const revision = s.revision;
    const old = s.upload; if (old) await discard(old);
    if (revision !== s.revision) return;
    const epoch = ++s.uploadEpoch;
    s.job = null; s.upload = null; s.uploading = true; video.pause(); notice('');
    if (s.previewURL) URL.revokeObjectURL(s.previewURL); s.previewURL = null;
    $('#creation-status').hidden = true; $('#upload-drop').hidden = true; $('#upload-summary').hidden = false;
    $('#upload-filename').textContent = file.name; $('#upload-detail').textContent = sizeLabel(file.size);
    $('#upload-progress').hidden = false; $('#upload-meter').value = 0; $('#upload-status').textContent = 'Uploading your clip…';
    $('#creator-canvas').hidden = true; $('#creator-result').hidden = true; $('#creator-stage-empty').hidden = false;
    updateControls();
    let token;
    try {
      token = await tokenFor(revision);
      const upload = await uploadRequest(file, token, epoch);
      if (epoch !== s.uploadEpoch || revision !== s.revision) {
        requestJSON(`/api/contributions/uploads/${upload.id}`, { base, method: 'DELETE', headers: { Authorization: `Bearer ${token}` } }).catch(() => {}); return;
      }
      s.upload = upload; s.start = 0; s.end = Math.min(6, Math.floor(upload.duration * 10) / 10); s.aspect = 'original'; s.rotation = 0;
      $('#crop-x').value = $('#crop-y').value = '0.5';
      for (const button of document.querySelectorAll('[data-aspect]')) { const selected = button.dataset.aspect === 'original'; button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', selected); }
      for (const id of ['trim-start', 'trim-end', 'start-seconds', 'end-seconds']) $(`#${id}`).max = String(Math.floor(upload.duration * 10) / 10);
      const blob = await media(upload.preview_url, revision);
      if (epoch !== s.uploadEpoch) return;
      s.previewURL = URL.createObjectURL(blob); video.src = s.previewURL; video.load();
      $('#creator-stage-empty').hidden = true; $('#creator-canvas').hidden = false;
      $('#upload-detail').textContent = `${upload.duration.toFixed(1)} s · ${sizeLabel(file.size)} · ready to make yours`;
      filmstrip(s.previewURL, epoch);
    } catch (error) {
      if (epoch !== s.uploadEpoch || error.name === 'AbortError') return;
      notice(error.message); $('#upload-drop').hidden = false; $('#upload-summary').hidden = true;
    } finally {
      if (epoch === s.uploadEpoch) { s.uploading = false; $('#upload-progress').hidden = true; updateControls(); }
    }
  }

  $('#creator-file').addEventListener('change', event => { const file = event.target.files[0]; event.target.value = ''; if (file) chooseFile(file); });
  $('#replace-video').addEventListener('click', () => $('#creator-file').click());
  $('#upload-drop').addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('#creator-file').click(); } });
  for (const name of ['dragenter', 'dragover']) $('#upload-drop').addEventListener(name, event => { event.preventDefault(); $('#upload-drop').classList.add('drag-over'); });
  for (const name of ['dragleave', 'drop']) $('#upload-drop').addEventListener(name, event => { event.preventDefault(); $('#upload-drop').classList.remove('drag-over'); if (name === 'drop' && event.dataTransfer.files[0]) chooseFile(event.dataTransfer.files[0]); });
  $('#cancel-upload').addEventListener('click', () => {
    ++s.uploadEpoch; s.uploading = false; s.upload = null;
    $('#upload-progress').hidden = true; $('#upload-summary').hidden = true; $('#upload-drop').hidden = false;
    toast('Upload cancelled.'); updateControls();
  });
  for (const id of ['creator-gif-title', 'creator-description']) $(`#${id}`).addEventListener('input', updateControls);

  async function sync(item, remove = false) {
    if (!s.user || !syncCloud) return;
    const revision = s.revision;
    const key = JSON.stringify([item.id, item.status, item.title, item.description, item.visibility, remove]);
    if (s.cloud.has(key)) return;
    s.cloud.set(key, 'pending');
    try {
      await syncCloud(item, remove, s.user.id);
      if (revision === s.revision) {
        s.cloud.set(key, 'done');
        if (remove) rememberCloudDelete(item.id, false);
      }
    } catch (error) {
      if (revision === s.revision) {
        s.cloud.set(key, 'failed'); s.cloudError = true; $('#contribution-cloud-note').hidden = true;
        logSystem(error, { level: 'error', context: 'cloud-sync', once: true });
        if (remove) rememberCloudDelete(item.id, true);
      }
    }
  }

  function pendingCloudDeletes() {
    try {
      const ids = JSON.parse(localStorage.getItem(`gifly:contribution-deletes:${s.user.id}`) || '[]');
      return Array.isArray(ids) ? ids.filter(id => typeof id === 'string' && /^contrib_[a-f0-9]{32}$/.test(id)).slice(0, 200) : [];
    }
    catch { return []; }
  }
  function rememberCloudDelete(id, pending) {
    if (!s.user) return;
    const ids = new Set(pendingCloudDeletes()); if (pending) ids.add(id); else ids.delete(id);
    try { localStorage.setItem(`gifly:contribution-deletes:${s.user.id}`, JSON.stringify([...ids])); } catch { /* Account data remains on the server. */ }
  }

  function schedulePoll() {
    clearTimeout(s.poll);
    if (s.user && (s.items.some(item => ACTIVE.has(item.status)) || s.job && ACTIVE.has(s.job.status))) s.poll = setTimeout(() => loadLibrary(true), 1700);
  }

  async function showJob(item) {
    const revision = s.revision;
    const previous = s.job; s.job = item; video.pause();
    $('#creator-gif-title').value = item.title; $('#creator-description').value = item.description;
    $('#creation-status').hidden = false; $('#creation-meter').value = item.progress;
    $('#cancel-creation').hidden = !ACTIVE.has(item.status);
    $('#retry-creation').hidden = !item.can_retry;
    $('#download-created').hidden = item.status !== 'ready';
    $('#share-created').hidden = item.status !== 'ready' || item.visibility !== 'public';
    $('#create-another').hidden = ACTIVE.has(item.status);
    $('#creator-gif-title').disabled = $('#creator-description').disabled = ACTIVE.has(item.status) || item.status === 'ready';
    for (const radio of document.querySelectorAll('[name="creator-visibility"]')) radio.disabled = ACTIVE.has(item.status) || item.status === 'ready';
    const messages = { queued: ['Your moment is in line…', 'You can keep browsing. Your GIF will appear in Your contribution.'],
      rendering: ['Making your little GIF…', 'Trimming, framing and finding the right colors.'],
      indexing: ['Giving your GIF a home…', 'Saving your description so your public GIF can be found.'],
      ready: ['A little GIF. Entirely yours.', `${sizeLabel(item.bytes)} · ${item.width} × ${item.height} · ${item.duration.toFixed(1)} s · ${item.visibility === 'public' ? 'Public and searchable' : 'Only you can see it'}`],
      failed: ['Creation stopped.', ''], cancelled: ['Creation cancelled.', 'Try again, or start with a different clip.'] };
    if (item.status === 'failed') logSystem(item.error, { level: 'error', context: `gif-creation:${item.id}`, once: true });
    const [heading, message] = messages[item.status] || ['Making your GIF…', ''];
    $('#creation-heading').textContent = heading; $('#creation-message').textContent = message;
    $('#creation-status').classList.toggle('is-ready', item.status === 'ready');
    $('#creation-status').classList.remove('is-error');
    if (item.status === 'ready' && previous?.status !== 'ready') {
      try {
        const url = await gifBlob(item);
        if (revision !== s.revision || s.job?.id !== item.id) return;
        $('#creator-result').src = url; $('#creator-result').hidden = false; $('#creator-canvas').hidden = true; $('#creator-stage-empty').hidden = true;
        toast('Your GIF is ready!');
      } catch (error) { if (revision !== s.revision) return; if (error.name !== 'AbortError') notice(error.message); }
    }
    if (revision !== s.revision) return;
    updateControls(); sync(item);
  }

  $('#creator-form').addEventListener('submit', async event => {
    event.preventDefault(); if (s.submitting || !s.upload || s.job) return;
    if (!s.user) { requestLogin(); return; }
    s.submitting = true; updateControls(); notice('');
    try {
      const item = await api('', { method: 'POST', body: { upload_id: s.upload.id,
        ...normalizeGIFDetails($('#creator-gif-title').value, $('#creator-description').value),
        visibility: document.querySelector('[name="creator-visibility"]:checked').value, ...settings() } });
      await showJob(item); await loadLibrary(true);
    } catch (error) { if (error.name !== 'AbortError') notice(error.message); }
    finally { s.submitting = false; updateControls(); }
  });

  async function gifBlob(item) {
    if (s.blobs.has(item.id)) return s.blobs.get(item.id);
    const blob = await media(item.url); const url = URL.createObjectURL(blob); s.blobs.set(item.id, url); return url;
  }

  async function download(item) {
    const revision = s.revision;
    try {
      const url = await gifBlob(item); if (revision !== s.revision) return;
      const a = document.createElement('a'); a.href = url;
      a.download = downloadFilename(item);
      document.body.append(a); a.click(); a.remove();
    } catch (error) { logSystem(error, { level: 'error', context: 'download' }); }
  }

  async function share(item) {
    const url = new URL(base + item.url, location.href).href;
    try { await navigator.clipboard.writeText(url); toast('Public GIF link copied.'); }
    catch (error) { logSystem(error, { level: 'error', context: 'clipboard' }); logSystem(`Public link: ${url}`, { context: 'clipboard' }); }
  }
  $('#download-created').addEventListener('click', () => s.job && download(s.job));
  $('#share-created').addEventListener('click', () => s.job && share(s.job));
  $('#cancel-creation').addEventListener('click', async () => {
    const button = $('#cancel-creation'); button.disabled = true;
    try { await api(`/${s.job.id}/cancel`, { method: 'POST' }); toast('Cancelling your GIF…'); schedulePoll(); }
    catch (error) { if (error.name !== 'AbortError') notice(error.message); }
    finally { button.disabled = false; }
  });
  $('#retry-creation').addEventListener('click', async () => {
    try { await showJob(await api(`/${s.job.id}/retry`, { method: 'POST' })); await loadLibrary(true); }
    catch (error) { if (error.name !== 'AbortError') notice(error.message); }
  });

  function resetCreator() {
    ++s.uploadEpoch; video.pause(); video.removeAttribute('src'); video.load();
    if (s.previewURL) URL.revokeObjectURL(s.previewURL); s.previewURL = null;
    s.upload = null; s.job = null; s.uploading = false; s.start = s.end = 0;
    $('#creator-form').reset(); $('#creator-gif-title').disabled = $('#creator-description').disabled = false;
    for (const radio of document.querySelectorAll('[name="creator-visibility"]')) radio.disabled = false;
    $('#upload-drop').hidden = false; $('#upload-summary').hidden = true; $('#upload-progress').hidden = true;
    $('#creation-status').hidden = true; $('#creator-stage-empty').hidden = false; $('#creator-canvas').hidden = true; $('#creator-result').hidden = true;
    notice(''); updateControls();
  }
  $('#create-another').addEventListener('click', () => { const old = s.upload; resetCreator(); discard(old); });

  function button(label, action, style = 'studio-text-button') {
    const element = document.createElement('button'); element.type = 'button'; element.className = style; element.textContent = label;
    element.addEventListener('click', action); return element;
  }

  async function toggleVisibility(item) {
    if (item.status !== 'ready' || s.visibilityRequests.has(item.id)) return;
    const revision = s.revision;
    s.visibilityRequests.add(item.id); renderLibrary(); notice('', '#contribution-notice');
    try {
      const updated = await api(`/${item.id}/visibility`, { method: 'PATCH',
        body: { visibility: item.visibility === 'public' ? 'private' : 'public' } });
      if (revision !== s.revision) return;
      s.items = s.items.map(current => current.id === updated.id ? updated : current);
      sync(updated);
      if (s.job?.id === updated.id) await showJob(updated);
      if (revision === s.revision) toast(updated.visibility === 'public' ? 'Your GIF is now public and searchable.' : 'Your GIF is now private. Only you can view it.');
    } catch (error) { if (revision === s.revision && error.name !== 'AbortError') notice(error.message, '#contribution-notice'); }
    finally {
      if (revision === s.revision) { s.visibilityRequests.delete(item.id); if (s.page === 'library') renderLibrary(); }
    }
  }

  function libraryCard(item) {
    const card = document.createElement('article'); card.className = 'contribution-card';
    const preview = document.createElement('div'); preview.className = 'contribution-card-preview';
    if (item.status === 'ready') {
      const image = document.createElement('img'); image.alt = item.description; image.loading = 'lazy'; image.decoding = 'async';
      gifBlob(item).then(url => { if (image.isConnected) image.src = url; }).catch(error => {
        if (image.isConnected) image.hidden = true;
        logSystem(error, { level: 'error', context: `gif-preview:${item.id}`, once: true });
      });
      preview.append(image);
    } else {
      const icon = document.createElement('span'); icon.className = 'card-status-icon'; icon.textContent = ACTIVE.has(item.status) ? '✳' : '↻';
      const status = document.createElement('span'); status.className = 'card-processing'; status.textContent = ACTIVE.has(item.status) ? `${item.status === 'queued' ? 'Waiting for your turn' : item.status === 'indexing' ? 'Saving your moment' : 'Creating your GIF'} · ${item.progress}%` : item.status === 'failed' ? 'Creation stopped' : 'Cancelled';
      preview.append(icon, status);
    }
    const badge = button(item.visibility === 'public' ? '◎ Public' : '♧ Private', () => toggleVisibility(item), `contribution-visibility ${item.visibility}`);
    badge.setAttribute('role', 'switch'); badge.setAttribute('aria-checked', item.visibility === 'public');
    badge.setAttribute('aria-label', `Make ${item.title} ${item.visibility === 'public' ? 'private' : 'public'}`);
    badge.title = item.status === 'ready' ? `Click to make ${item.visibility === 'public' ? 'private' : 'public'}` : 'Available when your GIF is ready';
    badge.disabled = item.status !== 'ready' || s.visibilityRequests.has(item.id);
    badge.setAttribute('aria-busy', s.visibilityRequests.has(item.id)); preview.append(badge);
    const body = document.createElement('div'); body.className = 'contribution-card-body';
    const title = document.createElement('h2'); title.textContent = item.title;
    const description = document.createElement('p'); description.textContent = item.description;
    const meta = document.createElement('span'); meta.className = 'contribution-card-meta';
    meta.textContent = item.status === 'ready' ? `${sizeLabel(item.bytes)} · ${item.duration.toFixed(1)} s · ${new Date(item.created * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}` : 'Your GIF will stay here when you leave this page.';
    if (item.status === 'failed') logSystem(item.error, { level: 'error', context: `gif-creation:${item.id}`, once: true });
    const actions = document.createElement('div'); actions.className = 'contribution-card-actions';
    if (item.status === 'ready') {
      actions.append(button('↓ Download', () => download(item)), button('Edit', () => edit(item)));
      if (item.visibility === 'public') actions.append(button('↗ Share', () => share(item)));
    } else if (item.can_retry) actions.append(button('↻ Try again', async () => {
      try { await api(`/${item.id}/retry`, { method: 'POST' }); await loadLibrary(); } catch (error) { if (error.name !== 'AbortError') notice(error.message, '#contribution-notice'); }
    }));
    if (ACTIVE.has(item.status)) actions.append(button('Cancel', async () => {
      try { await api(`/${item.id}/cancel`, { method: 'POST' }); schedulePoll(); } catch (error) { logSystem(error, { level: 'error', context: 'contributions' }); }
    }));
    else actions.append(button('Delete', () => confirmDelete(item), 'studio-text-button delete-link'));
    body.append(title, description, meta, actions); card.append(preview, body); return card;
  }

  function renderLibrary() {
    const items = s.items.filter(item => s.filter === 'all' || item.visibility === s.filter);
    $('#contribution-grid').replaceChildren(...items.map(libraryCard));
    $('#contribution-grid').setAttribute('aria-busy', s.loading);
    $('#contribution-count').textContent = s.user ? `${items.length} ${items.length === 1 ? 'little moment' : 'little moments'}` : 'Your personal collection';
    $('#contribution-empty').hidden = s.loading || items.length > 0;
    $('#contribution-cloud-note').hidden = true;
  }

  async function loadLibrary(silent = false) {
    if (!s.user || s.loading) return;
    const revision = s.revision; s.loading = true;
    if (!silent) { notice('', '#contribution-notice'); $('#contribution-grid').setAttribute('aria-busy', true); }
    try {
      const data = await api(); if (revision !== s.revision) return;
      const current = new Map(s.items.map(item => [item.id, item]));
      s.items = data.gifs.map(item => s.visibilityRequests.has(item.id) && current.has(item.id) ? current.get(item.id) : item);
      if (s.job) { const updated = s.items.find(item => item.id === s.job.id); if (updated) await showJob(updated); }
      for (const item of s.items) sync(item);
      for (const id of pendingCloudDeletes()) sync({ id }, true);
      if (s.page === 'library') renderLibrary();
    } catch (error) { if (revision === s.revision && error.name !== 'AbortError') notice(error.message, '#contribution-notice'); }
    finally { if (revision === s.revision) { s.loading = false; if (s.page === 'library') renderLibrary(); $('#contribution-grid').setAttribute('aria-busy', false); schedulePoll(); } }
  }

  function edit(item) {
    s.editing = item; $('#edit-contribution-title').value = item.title; $('#edit-contribution-description').value = item.description;
    $('#edit-contribution-visibility').value = item.visibility; $('#contribution-edit-error').hidden = true;
    $('#contribution-edit-dialog').showModal();
  }
  $('#contribution-edit-form').addEventListener('submit', async event => {
    event.preventDefault(); if (!s.editing) return;
    const submit = $('#save-contribution-edit'); submit.disabled = true;
    try {
      const updated = await api(`/${s.editing.id}`, { method: 'PUT', body: {
        ...normalizeGIFDetails($('#edit-contribution-title').value, $('#edit-contribution-description').value),
        visibility: $('#edit-contribution-visibility').value } });
      sync(updated); $('#contribution-edit-dialog').close(); toast('Your GIF details are saved.');
      if (s.job?.id === updated.id) await showJob(updated);
      await loadLibrary();
    } catch (error) { logSystem(error, { level: 'error', context: 'contribution-edit' }); }
    finally { submit.disabled = false; }
  });

  function confirmDelete(item) {
    s.deleting = item; $('#contribution-delete-description').textContent = `“${item.title}”`;
    $('#contribution-delete-error').hidden = true; $('#contribution-delete-dialog').showModal();
  }
  $('#keep-contribution').addEventListener('click', () => $('#contribution-delete-dialog').close());
  $('#confirm-contribution-delete').addEventListener('click', async () => {
    if (!s.deleting) return; const item = s.deleting; const submit = $('#confirm-contribution-delete'); submit.disabled = true;
    try {
      await api(`/${item.id}`, { method: 'DELETE' }); sync(item, true);
      const url = s.blobs.get(item.id); if (url) URL.revokeObjectURL(url); s.blobs.delete(item.id);
      if (s.job?.id === item.id) resetCreator();
      $('#contribution-delete-dialog').close(); toast('Your GIF has been deleted.'); await loadLibrary();
    } catch (error) { logSystem(error, { level: 'error', context: 'contribution-delete' }); }
    finally { submit.disabled = false; }
  });
  $('#retry-cloud-sync').addEventListener('click', async () => {
    s.cloud.clear(); s.cloudError = false; $('#retry-cloud-sync').disabled = true;
    await Promise.all(s.items.map(item => sync(item)));
    await Promise.all(pendingCloudDeletes().map(id => sync({ id }, true)));
    $('#retry-cloud-sync').disabled = false; $('#contribution-cloud-note').hidden = true;
    if (!s.cloudError) toast('Your contribution is synced to your account.');
  });
  document.querySelectorAll('[data-library-filter]').forEach(element => element.addEventListener('click', () => {
    s.filter = element.dataset.libraryFilter;
    for (const option of document.querySelectorAll('[data-library-filter]')) { const selected = option === element; option.classList.toggle('selected', selected); option.setAttribute('aria-pressed', selected); }
    renderLibrary();
  }));
  for (const id of ['create-nav', 'library-create', 'empty-create']) $(`#${id}`).addEventListener('click', () => show('creator'));
  for (const id of ['contributions-nav', 'view-contributions']) $(`#${id}`).addEventListener('click', () => show('library'));
  updateControls();

  return {
    isExplore: () => s.page === 'explore',
    showExplore: () => show('explore'),
    setAccount(user) {
      if (s.user?.id === user?.id) { s.user = user; return; }
      const previous = s.user; s.user = user; ++s.revision;
      clearTimeout(s.poll); s.items = []; s.cloud.clear(); s.cloudError = false; s.loading = false;
      s.editing = null; s.deleting = null; $('#contribution-grid').replaceChildren();
      s.visibilityRequests.clear();
      revokeMedia();
      for (const id of ['contribution-edit-dialog', 'contribution-delete-dialog']) $(`#${id}`).close();
      if (previous) { resetCreator(); s.pendingFile = null; }
      if (user) {
        if (s.afterLogin === 'library') { s.afterLogin = null; show('library'); }
        else if (s.pendingFile) { const file = s.pendingFile; s.pendingFile = null; s.afterLogin = null; chooseFile(file); }
        else if (s.page === 'library') loadLibrary();
      } else if (s.page === 'library') { renderLibrary(); notice('Log in to see the GIFs you created.', '#contribution-notice'); }
    },
  };
}
