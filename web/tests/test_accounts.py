"""The retained upstream SQLite store is tested independently of Firebase HTTP auth."""
import tempfile
import time
import unittest
from pathlib import Path
from web.accounts import AccountStore


class LegacyAccountStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'accounts.sqlite3'
        self.store = AccountStore(self.path)

    def test_password_hashes_case_insensitive_login_and_duplicates(self):
        alice = self.store.register('Alice', 'correct horse battery')
        self.assertEqual(self.store.login('ALICE', 'correct horse battery'), alice)
        self.assertIsNone(self.store.login('Alice', 'wrong password'))
        with self.assertRaises(ValueError):
            self.store.register('alice', 'another password')
        with self.store.connect() as db:
            self.assertNotEqual(db.execute('SELECT password_hash FROM users').fetchone()[0], 'correct horse battery')

    def test_favorites_are_durable_and_isolated(self):
        alice = self.store.register('Alice', 'correct horse battery')
        bob = self.store.register('Bob', 'another good password')
        self.store.save(alice['id'], 'happy', True)
        self.store.save(alice['id'], 'happy', True)
        reopened = AccountStore(self.path)
        self.assertEqual(reopened.saved(alice['id']), ['happy'])
        self.assertEqual(reopened.saved(bob['id']), [])
        reopened.save(bob['id'], 'happy', False)
        self.assertEqual(reopened.saved(alice['id']), ['happy'])

    def test_session_expiration_logout_and_rotation(self):
        user = self.store.register('Alice', 'correct horse battery')
        token = self.store.new_session(user['id'])
        self.assertEqual(self.store.user_for(token), user)
        replacement = self.store.new_session(user['id'], previous=token)
        self.assertIsNone(self.store.user_for(token))
        with self.store.connect() as db:
            self.assertNotEqual(db.execute('SELECT token_hash FROM sessions').fetchone()[0], replacement)
            db.execute('UPDATE sessions SET expires=?', (int(time.time()) - 1,))
        self.assertIsNone(self.store.user_for(replacement))
        token = self.store.new_session(user['id'])
        self.store.logout(token)
        self.assertIsNone(self.store.user_for(token))


if __name__ == '__main__':
    unittest.main()
