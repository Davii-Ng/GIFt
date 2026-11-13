"""Real video conversion, durable ownership, and public/private retrieval boundaries."""
import io
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from web.contribution_auth import current_contributor
from web.contribution_media import CREATE_FLAGS, MediaError, geometry, probe_video
from web.contributions import ContributionService, CreateGIF, UpdateGIF, router
from web.gif_downloads import router as downloads_router


class TestModel:
    def encode(self, text, **kwargs):
        text = text.lower()
        return np.array([1., 0., 0.]) if 'happy' in text else np.array([0., 1., 0.]) if 'sad' in text else np.array([0., 0., 1.])


class ContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.media_dir.cleanup)
        cls.video = Path(cls.media_dir.name) / 'happy.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
                        '-i', 'testsrc2=size=1280x720:rate=12', '-t', '1.5', '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', str(cls.video)], check=True, creationflags=CREATE_FLAGS)
        cls.video_bytes = cls.video.read_bytes()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ContributionService(self.directory.name, model_factory=TestModel)
        self.addCleanup(self.store.close)
        self.app = FastAPI()
        self.app.state.contributions = self.store
        self.app.include_router(router)
        self.app.include_router(downloads_router)
        self.app.state.by_id = {}
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.project_patch = patch('web.contribution_auth.firebase_project', return_value='test-project')
        self.project_patch.start(); self.addCleanup(self.project_patch.stop)
        self.verify_patch = patch('web.contribution_auth.verify_firebase_token', side_effect=lambda token, *a, **k:
                                  {'sub': token, 'iss': 'https://securetoken.google.com/test-project'})
        self.verify_patch.start(); self.addCleanup(self.verify_patch.stop)
        self.headers = {'Authorization': 'Bearer alice'}

    def upload(self, owner='alice'):
        response = self.client.post('/api/contributions/uploads', files={'file': ('clip.mp4', self.video_bytes, 'video/mp4')},
                                    headers={'Authorization': f'Bearer {owner}'})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def create(self, upload, visibility='private', **changes):
        body = dict(upload_id=upload['id'], title='My little moment', description='A happy dance',
                    visibility=visibility, start=0, end=1, aspect='original', rotation=0)
        body.update(changes)
        response = self.client.post('/api/contributions', json=body, headers=self.headers)
        self.assertEqual(response.status_code, 202, response.text)
        gif_id = response.json()['id']
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            item = self.client.get(f'/api/contributions/{gif_id}', headers=self.headers).json()
            if item['status'] not in ('queued', 'rendering', 'indexing'):
                self.assertEqual(item['status'], 'ready', item)
                return item
            time.sleep(.1)
        self.fail('GIF creation timed out')

    def test_real_conversion_and_private_owner_isolation(self):
        self.assertEqual(self.client.get('/api/contributions').status_code, 401)
        self.assertEqual(self.client.post('/api/contributions/uploads', files={'file': ('a.mp4', b'bad')}).status_code, 401)
        upload = self.upload()
        self.assertEqual(self.client.get(upload['preview_url']).status_code, 401)
        self.assertEqual(self.client.get(upload['preview_url'], headers={'Authorization': 'Bearer bob'}).status_code, 404)
        preview = self.client.get(upload['preview_url'], headers=self.headers)
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.headers['content-type'], 'video/mp4')
        item = self.create(upload)
        self.assertEqual((item['width'], item['height']), (854, 480))
        self.assertTrue(item['bytes'] > 0)
        self.assertEqual(self.client.get(item['url']).status_code, 404)
        self.assertEqual(self.client.get(item['url'], headers={'Authorization': 'Bearer bob'}).status_code, 404)
        output = self.client.get(item['url'], headers=self.headers)
        self.assertEqual(output.status_code, 200)
        self.assertTrue(output.content.startswith((b'GIF89a', b'GIF87a')))
        self.assertIn(b'NETSCAPE2.0', output.content)
        self.assertEqual(output.headers['cache-control'], 'no-store')
        self.assertEqual(self.store.public(), [])
        self.assertEqual(self.store.search('happy'), [])
        self.assertEqual(self.client.get('/api/contributions', headers={'Authorization': 'Bearer bob'}).json()['gifs'], [])
        self.assertEqual(self.client.get(f"/api/contributions/{item['id']}", headers={'Authorization': 'Bearer bob'}).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/contributions/{item['id']}", headers={'Authorization': 'Bearer bob'}).status_code, 404)
        self.assertFalse(self.store.source_path({'id': upload['id'], 'suffix': '.mp4'}).exists())

    def test_public_search_description_update_unpublish_and_delete(self):
        item = self.create(self.upload(), 'public')
        self.assertEqual(self.client.get(item['url']).status_code, 200)
        self.assertEqual(self.store.search('happy')[0]['id'], item['id'])
        self.assertGreater(self.store.search('happy')[0]['score'], .5)
        self.assertEqual(self.store.search('happy', hidden=[item['id']]), [])
        self.assertEqual(self.store.search('happy', category='animals'), [])
        response = self.client.put(f"/api/contributions/{item['id']}", headers=self.headers,
                                  json={'title': 'Another mood', 'description': 'A sad reaction', 'visibility': 'public'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(self.store.search('sad')[0]['score'], .5)
        self.assertAlmostEqual(self.store.search('happy')[0]['score'], 0)
        response = self.client.put(f"/api/contributions/{item['id']}", headers=self.headers,
                                  json={'title': 'Just mine', 'description': 'A sad reaction', 'visibility': 'private'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get(item['url']).status_code, 404)
        self.assertEqual(self.store.public(), [])
        self.assertEqual(self.store.search('sad'), [])
        # Even a stale vector must never override canonical privacy.
        self.store.index().upsert(ids=[item['id']], embeddings=[[0., 1., 0.]], documents=['sad'])
        self.assertEqual(self.store.search('sad'), [])
        self.assertEqual(self.client.delete(f"/api/contributions/{item['id']}", headers=self.headers).status_code, 204)
        self.assertFalse(self.store.file_path(item['id']).exists())
        self.assertEqual(self.client.get(item['url']).status_code, 404)

    def test_validation_crop_and_upload_boundaries(self):
        self.assertEqual(self.client.post('/api/contributions/uploads', files={'file': ('a.mp4', b'not a video')}, headers=self.headers).status_code, 422)
        self.assertEqual(self.client.post('/api/contributions/uploads', files={'file': ('a.html', b'html')}, headers=self.headers).status_code, 415)
        self.assertEqual(list((self.store.root / 'uploads').iterdir()), [])
        with self.assertRaises(ValueError):
            CreateGIF(upload_id='0' * 32, title='x', description='valid', start=0, end=11)
        with self.assertRaises(ValueError):
            CreateGIF(upload_id='0' * 32, title=' ', description='   ', start=0, end=1)
        with self.assertRaises(ValueError):
            CreateGIF(upload_id='0' * 32, title='x', description='valid', start=float('nan'), end=1)
        upload = self.upload()
        item = self.create(upload, aspect='square', rotation=90)
        self.assertEqual((item['width'], item['height']), (480, 480))
        duplicate = self.client.post('/api/contributions', json=dict(upload_id=upload['id'], title='x', description='happy', end=1), headers=self.headers)
        self.assertIn(duplicate.status_code, (404, 409))

    def test_cancel_retry_and_durable_library(self):
        upload = self.upload()
        with patch.object(self.store, 'process'):
            item = self.store.create('alice', CreateGIF(upload_id=upload['id'], title='Test', description='happy', end=1))
            self.store.cancel(item['id'], 'alice')
            self.assertTrue(self.store.cancellations[item['id']].is_set())
        # Simulate a stopped process and reopen the persistent database.
        self.store.cancellations.pop(item['id'], None)
        reopened = ContributionService(self.directory.name, model_factory=TestModel)
        self.addCleanup(reopened.close)
        existing = reopened.own('alice')[0]
        self.assertEqual(existing['status'], 'failed')
        self.assertTrue(existing['can_retry'])
        self.assertEqual(reopened.own('bob'), [])
        retry = reopened.retry(item['id'], 'alice')
        self.assertIn(retry['status'], ('queued','rendering'))
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline and reopened.get(item['id'])['status'] in ('queued','rendering','indexing'):
            time.sleep(.1)
        self.assertEqual(reopened.get(item['id'])['status'], 'ready')

    def test_token_issuer_and_owner_validation(self):
        with patch('web.contribution_auth.verify_firebase_token', return_value={'sub': 'alice', 'iss': 'https://wrong.example'}):
            self.assertEqual(self.client.get('/api/contributions', headers=self.headers).status_code, 401)
        with patch('web.contribution_auth.verify_firebase_token', return_value={'sub': '', 'iss': 'https://securetoken.google.com/test-project'}):
            self.assertEqual(self.client.get('/api/contributions', headers=self.headers).status_code, 401)

    def test_unpublishing_after_restart_removes_persisted_vector(self):
        item = self.create(self.upload(), 'public')
        reopened = ContributionService(self.directory.name, model_factory=TestModel)
        self.addCleanup(reopened.close)
        self.assertIsNone(reopened.collection)
        reopened.edit(item['id'], 'alice', UpdateGIF(title=item['title'], description=item['description'], visibility='private'))
        self.assertEqual(reopened.index().count(), 0)
        self.assertEqual(self.client.get(item['url']).status_code, 404)

    def test_expired_uploads_are_removed_before_preview_or_retry(self):
        upload = self.upload()
        with self.store.db() as db:
            db.execute('UPDATE uploads SET created=? WHERE id=?', (time.time() - 25 * 3600, upload['id']))
        response = self.client.get(upload['preview_url'], headers=self.headers)
        self.assertEqual(response.status_code, 404)
        self.assertFalse(self.store.preview_path(upload['id']).exists())
        self.assertFalse(self.store.source_path({'id': upload['id'], 'suffix': '.mp4'}).exists())

    def test_create_with_only_name_or_only_description_and_edit(self):
        for details, expected in [({'title': 'X', 'description': ''}, 'X'),
                                  ({'title': '', 'description': 'A happy dance ' * 15}, ('A happy dance ' * 15).strip())]:
            with self.subTest(details=details):
                item = self.create(self.upload(), 'public', **details)
                self.assertEqual(item['title'], expected)
                self.assertEqual(item['description'], expected)
                updated = self.client.put(f"/api/contributions/{item['id']}", headers=self.headers,
                                          json={'title': 'sad', 'visibility': 'public'})
                self.assertEqual(updated.status_code, 200, updated.text)
                self.assertEqual(updated.json()['description'], 'sad')
                self.assertEqual(self.store.search('sad')[0]['match_priority'], 3)
                rejected = self.client.put(f"/api/contributions/{item['id']}", headers=self.headers,
                                          json={'title': ' ', 'description': ' ', 'visibility': 'public'})
                self.assertEqual(rejected.status_code, 422)

    def test_visibility_switch_and_generic_download_enforce_owner(self):
        item = self.create(self.upload())
        key = item['id']
        before = self.store.get(key)
        path = f'/api/gifs/{key}/download'
        response = self.client.get(path, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment', response.headers['content-disposition'])
        self.assertTrue(response.content.startswith(b'GIF'))
        self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.get(path, headers={'Authorization': 'Bearer bob'}).status_code, 404)
        change = f'/api/contributions/{key}/visibility'
        self.assertEqual(self.client.patch(change, json={'visibility': 'public'}).status_code, 401)
        self.assertEqual(self.client.patch(change, json={'visibility': 'public'}, headers={'Authorization': 'Bearer bob'}).status_code, 404)
        response = self.client.patch(change, json={'visibility': 'public'}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['visibility'], 'public')
        self.assertEqual(self.client.get(path).status_code, 200)
        self.assertTrue(self.store.search('happy'))
        response = self.client.patch(change, json={'visibility': 'private'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.store.search('happy'), [])
        after = self.store.get(key)
        for field in ('title','description','vector','vector_version'):
            self.assertEqual(after[field], before[field])
        self.assertEqual(self.client.patch(change, json={'visibility': 'everyone'}, headers=self.headers).status_code, 422)


if __name__ == '__main__':
    unittest.main()
