import test from 'node:test';
import assert from 'node:assert/strict';
import { logSystem, consoleOnlyValidation } from '../system-log.js';

test('system errors log a scoped message without sensitive object details', context => {
  const output = context.mock.method(console, 'error', () => {});
  const error = Object.assign(new Error('Search temporarily unavailable.'), {
    token: 'private-token', request: { password: 'private-password' }, account: { email: 'private-email' },
  });
  logSystem(error, { level: 'error', context: 'search' });
  assert.deepEqual(output.mock.calls[0].arguments, ['[gifly:search] Search temporarily unavailable.']);
});

test('success and warning notifications go to their console levels', context => {
  const info = context.mock.method(console, 'info', () => {});
  const warning = context.mock.method(console, 'warn', () => {});
  logSystem('Your GIF is now public.');
  logSystem('Cloud sync unavailable.', { level: 'warn', context: 'cloud-sync' });
  assert.deepEqual(info.mock.calls[0].arguments, ['[gifly:app] Your GIF is now public.']);
  assert.deepEqual(warning.mock.calls[0].arguments, ['[gifly:cloud-sync] Cloud sync unavailable.']);
});

test('empty messages and cancelled requests remain silent', context => {
  const output = ['info', 'warn', 'error'].map(level => context.mock.method(console, level, () => {}));
  for (const message of ['', ' ', null, undefined, new DOMException('Account changed.', 'AbortError')]) {
    logSystem(message, { level: 'error' });
  }
  assert.ok(output.every(method => method.mock.calls.length === 0));
});

test('polling logs repeated failures once without suppressing other operations', context => {
  const output = context.mock.method(console, 'error', () => {});
  logSystem('Render failed.', { level: 'error', context: 'job-test-a', once: true });
  logSystem('Render failed.', { level: 'error', context: 'job-test-a', once: true });
  logSystem('Render failed.', { level: 'error', context: 'job-test-b', once: true });
  logSystem('Render failed.', { level: 'error', context: 'job-test-a' });
  assert.equal(output.mock.calls.length, 3);
});

test('native invalid fields retain validation but suppress message bubbles and values in logs', context => {
  const output = context.mock.method(console, 'warn', () => {});
  let listener;
  consoleOnlyValidation({ addEventListener(event, handler, capture) {
    assert.equal(event, 'invalid'); assert.equal(capture, true); listener = handler;
  } });
  let prevented = false;
  listener({ target: { validationMessage: 'Please fill out this field.', value: 'secret-value' },
    preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.deepEqual(output.mock.calls[0].arguments, ['[gifly:validation] Please fill out this field.']);
});
