import test from 'node:test';
import assert from 'node:assert/strict';
import { usernameIdentity, accountError } from '../firebase-account.js';
test('username identity is shared and case-insensitive', () => {
  assert.deepEqual(usernameIdentity(' User_12 '), usernameIdentity('user_12'));
  for (const name of ['ab', 'a.b', 'a-b', 'a b', 'x'.repeat(25)]) assert.throws(() => usernameIdentity(name));
});
test('credential errors hide implementation details', () => {
  assert.equal(accountError({code:'auth/invalid-credential'}), 'Tên đăng nhập hoặc mật khẩu không đúng.');
  assert.equal(accountError({code:'auth/wrong-password'}), accountError({code:'auth/user-not-found'}));
  assert.ok(!accountError({message:'secret-token'}).includes('secret-token'));
});

test('Firebase setup errors provide the missing setup step', () => {
  assert.match(accountError({code:'auth/configuration-not-found'}), /Authentication/);
  assert.match(accountError({code:'auth/operation-not-allowed'}), /Email\/Password/);
});
