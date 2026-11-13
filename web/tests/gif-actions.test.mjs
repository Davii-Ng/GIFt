import test from 'node:test';
import assert from 'node:assert/strict';
import { downloadFilename, gifDownloadPath, fetchGIFBlob } from '../gif-actions.js';

test('downloads use known IDs and resolve favorites from alternate search indexes', () => {
  assert.equal(gifDownloadPath({ id: 'contrib_abc' }), '/api/gifs/contrib_abc/download');
  assert.equal(gifDownloadPath({ id: 'gemini', favorite_id: 'catalog' }), '/api/gifs/catalog/download');
  assert.equal(gifDownloadPath({ id: 'id/with space' }), '/api/gifs/id%2Fwith%20space/download');
  assert.throws(() => gifDownloadPath({}), /not available/);
});

test('download filenames preserve readable Unicode and exclude unsafe path characters', () => {
  assert.equal(downloadFilename({ title: 'Một chút vui' }), 'Một chút vui.gif');
  assert.ok(!/[<>:"/\\|?*]/.test(downloadFilename({ title: '../a<>:"/\\|?*' })));
  assert.equal(downloadFilename({ title: 'CON' }), 'gif-CON.gif');
  assert.equal(downloadFilename({ title: 'x'.repeat(1000) }).length, 104);
});

test('GIF downloads pass the selected backend and optional owner token', async () => {
  let called;
  const blob = await fetchGIFBlob({ id: 'known' }, { base: 'http://127.0.0.1:8000', token: 'test-only',
    fetchImpl: async (url, options) => { called = { url, options }; return new Response('GIF89a-test'); } });
  assert.equal(called.url, 'http://127.0.0.1:8000/api/gifs/known/download');
  assert.equal(called.options.headers.Authorization, 'Bearer test-only');
  assert.equal(await blob.text(), 'GIF89a-test');
});

test('failed sources give readable errors and HTML is never downloaded as a GIF', async () => {
  await assert.rejects(fetchGIFBlob({ id: 'known' }, { fetchImpl: async () => new Response(JSON.stringify({ detail: 'Source unavailable' }), { status: 502 }) }), /Source unavailable/);
  await assert.rejects(fetchGIFBlob({ id: 'known' }, { fetchImpl: async () => new Response('<html>not a gif</html>') }), /valid GIF/);
});
