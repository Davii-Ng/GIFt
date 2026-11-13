export function downloadFilename(gif) {
  let name = [...String(gif.title || gif.description || 'gif').replace(/[<>:"/\\|?*\u0000-\u001f]/g, '').trim()].slice(0, 100).join('').replace(/[. ]+$/, '') || 'gif';
  if (/^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i.test(name)) name = 'gif-' + name;
  return name + '.gif';
}

export function gifDownloadPath(gif) {
  const id = gif.favorite_id || gif.id;
  if (!id) throw new Error('This GIF is not available for download.');
  return `/api/gifs/${encodeURIComponent(id)}/download`;
}

export async function fetchGIFBlob(gif, { base = '', token, fetchImpl = globalThis.fetch } = {}) {
  let response;
  try {
    response = await fetchImpl(base + gifDownloadPath(gif), { credentials: 'include',
      headers: token ? { Authorization: `Bearer ${token}` } : {}, cache: 'no-store' });
  } catch { throw new Error('Cannot reach the GIF server. Check your connection and try again.'); }
  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; } catch { /* Never show remote HTML as an error. */ }
    throw new Error(typeof detail === 'string' ? detail : 'This GIF could not be downloaded. Please try again.');
  }
  const blob = await response.blob();
  const signature = new TextDecoder().decode(await blob.slice(0, 6).arrayBuffer());
  if (!['GIF87a', 'GIF89a'].includes(signature)) throw new Error('The source did not return a valid GIF.');
  return blob;
}

export function saveGIFBlob(blob, gif) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = downloadFilename(gif);
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
