import test from 'node:test';
import assert from 'node:assert/strict';
import { cropGeometry, validateVideoFile, normalizeGIFDetails } from '../contributions.js';

test('preview geometry matches 480p output, crop positioning, and rotation', () => {
  assert.deepEqual([cropGeometry(1920, 1080, 'original').outputWidth, cropGeometry(1920, 1080, 'original').outputHeight], [854, 480]);
  const square = cropGeometry(1920, 1080, 'square', 0, 1, 0.5);
  assert.equal(square.x, 840);
  assert.equal(square.cropWidth, 1080);
  assert.equal(square.outputWidth, 480);
  const portrait = cropGeometry(1920, 1080, 'original', 90);
  assert.equal(portrait.outputWidth, 270);
  assert.equal(portrait.outputHeight, 480);
  const small = cropGeometry(320, 240, 'original');
  assert.equal(small.outputWidth, 320);
  assert.equal(small.outputHeight, 240);
});

test('video selection rejects empty, oversized, and unrelated files', () => {
  assert.equal(validateVideoFile({ name: 'my clip.MOV', size: 500 }), '');
  assert.match(validateVideoFile({ name: 'fake.mp4', size: 0 }), /empty/);
  assert.match(validateVideoFile({ name: 'photo.jpg', size: 500 }), /MP4/);
  assert.match(validateVideoFile({ name: 'large.mp4', size: 51 * 1024 * 1024 }), /50 MB/);
});

test('either metadata field can stand alone and is copied in full', () => {
  assert.deepEqual(normalizeGIFDetails('  Hello  ', ''), { title: 'Hello', description: 'Hello' });
  const description = 'A happy moment '.repeat(20).trim();
  assert.deepEqual(normalizeGIFDetails('', description), { title: description, description });
  assert.deepEqual(normalizeGIFDetails('X', ''), { title: 'X', description: 'X' });
  assert.deepEqual(normalizeGIFDetails('One', 'Two'), { title: 'One', description: 'Two' });
  assert.throws(() => normalizeGIFDetails(' ', '\n'), /name or a description/);
  assert.throws(() => normalizeGIFDetails('a'.repeat(1001), ''), /1000/);
});
