const LOCAL_API = 'http://127.0.0.1:8000';

export function apiBaseFor(location) {
  const isLocal = ['localhost', '127.0.0.1', '[::1]', '::1'].includes(location.hostname);
  const isLiveServer = ['5500', '5501'].includes(location.port);
  return location.protocol === 'http:' && isLocal && isLiveServer ? `http://${location.hostname}:8000` : '';
}

export class APIError extends Error {
  constructor(message, code, status = 0) {
    super(message); this.name = 'APIError'; this.code = code; this.status = status;
  }
}

export async function requestJSON(path, { base = '', signal, method = 'GET', body, headers = {}, fetchImpl = globalThis.fetch } = {}) {
  let response;
  try {
    response = await fetchImpl(`${base}${path}`, {
      signal, method, credentials: 'include',
      headers: { Accept: 'application/json', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...headers },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new APIError(`Cannot connect to the GIF server. Run start-web.ps1 and open ${LOCAL_API}.`, 'connection');
  }
  if (response.status === 204) return null;
  const contentType = (response.headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
  if (contentType !== 'application/json' && !contentType.endsWith('+json')) {
    throw new APIError(`This address returned a web page instead of GIF data. Run start-web.ps1 and open ${LOCAL_API}.`, 'invalid-response', response.status);
  }
  let data;
  try { data = await response.json(); }
  catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new APIError('The GIF server returned invalid data. Please try again.', 'invalid-response', response.status);
  }
  if (!response.ok) {
    const detail = typeof data?.detail === 'string' ? data.detail :
      Array.isArray(data?.detail) ? data.detail.map(item => `${item.loc?.at(-1) || 'Input'}: ${item.msg}`).join(' ') :
      `Could not load data (HTTP ${response.status}). Please try again.`;
    throw new APIError(detail, 'http', response.status);
  }
  return data;
}
