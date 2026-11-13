"""Exercise the real Chroma boundary after Rocchio feedback, with both GIF sources."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import chromadb
import numpy as np
from chromadb.config import Settings
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gift.gemini.feedback import normalize
from web.auto_search import AutoSearch
from web.contributions import ContributionService
from web.tools import local_vector, router


class FeedbackModel:
    def encode(self, text, **kwargs):
        if isinstance(text, list):
            return np.stack([self.encode(item, **kwargs) for item in text])
        return normalize([0.3, 0.8, 0.1])


class FeedbackIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client_db = chromadb.PersistentClient(path=str(Path(self.temp.name) / 'base'),
                                                 settings=Settings(anonymized_telemetry=False))
        self.addCleanup(self.client_db.close)
        self.base = self.client_db.create_collection('base_gifs', embedding_function=None,
                                                     configuration={'hnsw': {'space': 'cosine'}})
        self.rows = [dict(id=key, title=key, description=description, category='general',
                          url=f'https://example.com/{key}.gif', source='Sample')
                     for key, description in [('dance', 'a happy person dancing'),
                                              ('cat', 'a cat looking confused'),
                                              ('sad', 'someone looking sad')]]
        self.base.add(ids=[row['id'] for row in self.rows],
                      embeddings=[normalize(row).tolist() for row in ([0.1, 1., 0.1], [1., 0.1, 0.], [0., 0., 1.])],
                      documents=[row['description'] for row in self.rows],
                      metadatas=[{key: value for key, value in row.items() if key != 'id'} for row in self.rows])
        self.store = ContributionService(Path(self.temp.name) / 'community', model_factory=FeedbackModel)
        self.addCleanup(self.store.close)
        now = time.time()
        self.public_id = 'contrib_' + 'a' * 32
        self.private_id = 'contrib_' + 'b' * 32
        for key, visibility, vector in [(self.public_id, 'public', [0.2, 0.9, 0.1]),
                                        (self.private_id, 'private', [0.9, 0.1, 0.])]:
            values = normalize(vector).tolist()
            with self.store.db() as db:
                db.execute('''INSERT INTO contributions
                    (id,owner,upload_id,title,description,visibility,status,settings,vector,created,updated)
                    VALUES (?,?,?,?,?,?,'ready','{}',?,?,?)''',
                    (key, 'test-owner', '0' * 32, 'My happy dance', 'a happy person dancing', visibility,
                     json.dumps(values), now, now))
            if visibility == 'public':
                self.store.publish_vector(self.store.get(key), values)
        self.app = FastAPI()
        self.app.state.retriever = SimpleNamespace(collection=self.base, model=FeedbackModel())
        self.app.state.auto_search = AutoSearch(self.rows)
        self.app.state.contributions = self.store
        self.app.state.id_by_url = {}
        self.app.include_router(router)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.addCleanup(local_vector.cache_clear)

    def search(self, **changes):
        response = self.client.post('/api/tools/feedback', json=dict(query='a happy person dancing', model='auto', limit=12, **changes))
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertTrue(data['results'])
        self.assertTrue(all(np.isfinite(row['score']) for row in data['results']))
        self.assertNotIn(self.private_id, [row['id'] for row in data['results']])
        return data

    def test_search_without_votes_includes_both_sources(self):
        ids = [row['id'] for row in self.search()['results']]
        self.assertIn('dance', ids)
        self.assertIn(self.public_id, ids)

    def test_base_likes_and_dislikes_work_with_public_contributions(self):
        original = {row['id']: row['score'] for row in self.search()['results']}
        for vote in ('liked', 'disliked'):
            with self.subTest(vote=vote):
                data = self.search(**{vote: ['cat']})
                self.assertIn(self.public_id, [row['id'] for row in data['results']])
                scores = {row['id']: row['score'] for row in data['results']}
                if vote == 'liked':
                    self.assertGreater(scores['cat'], original['cat'])
                else:
                    self.assertLess(scores['cat'], original['cat'])

    def test_public_contribution_likes_dislikes_and_mixed_votes(self):
        for changes in ({'liked': [self.public_id]}, {'disliked': [self.public_id]},
                        {'liked': ['dance', self.public_id], 'disliked': ['cat']}):
            with self.subTest(changes=changes):
                self.search(**changes)

    def test_dislike_memory_reset_and_restore(self):
        original = self.search()
        memory = [{'gif_id': self.public_id, 'query': 'a happy person dancing', 'query_vector': original['vector']}]
        data = self.search(memory=memory)
        self.assertNotIn(self.public_id, [row['id'] for row in data['results']])
        self.assertEqual(data['hidden'], 1)
        restored = self.search(memory=memory, unhide=True)
        self.assertIn(self.public_id, [row['id'] for row in restored['results']])
        self.assertEqual(restored['memory'], [])
        self.search(liked=[self.private_id, 'deleted-gif'])
        self.search()  # Resetting votes retains normal search.

    def test_optional_community_failure_does_not_break_existing_search(self):
        with patch.object(self.store, 'search', side_effect=ValueError('Broken optional index')), self.assertLogs('gifly.tools', level='ERROR'):
            data = self.search(liked=['dance'])
            self.assertIn('dance', [row['id'] for row in data['results']])
        with patch.object(self.store, 'feedback_vectors', side_effect=ValueError('Broken optional vectors')), self.assertLogs('gifly.tools', level='ERROR'):
            data = self.search(liked=['dance', self.public_id])
            self.assertIn('dance', [row['id'] for row in data['results']])

    def test_numpy_feedback_vector_can_query_community_directly(self):
        hits = self.store.search('happy', vector=normalize([0.3, 0.8, 0.1]))
        self.assertEqual(hits[0]['id'], self.public_id)

    def test_relevant_community_keyword_match_remains_prominent(self):
        data = self.search()
        self.assertEqual(data['results'][0]['id'], self.public_id)
        self.assertEqual(data['results'][0]['score'], data['results'][0]['similarity'])

    def test_names_and_descriptions_are_both_keyword_searchable(self):
        with self.store.db() as db:
            db.execute('UPDATE contributions SET title=? WHERE id=?', ('Purple Moonwalk', self.public_id))
        for query in ('PURPLE MOONWALK', 'purple', 'purple dancing', '"Purple Moonwalk"', '"happy person"'):
            with self.subTest(query=query):
                response = self.client.post('/api/tools/feedback', json={'query': query, 'model': 'auto', 'limit': 1})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['results'][0]['id'], self.public_id)
        with self.store.db() as db:
            db.execute('UPDATE contributions SET title=? WHERE id=?', ('Private Secret', self.private_id))
        response = self.client.post('/api/tools/feedback', json={'query': '"Private Secret"', 'model': 'auto'})
        self.assertNotIn(self.private_id, [row['id'] for row in response.json()['results']])

    def test_exact_name_is_found_outside_vector_top_k_and_respects_hidden(self):
        needle = 'contrib_' + 'c' * 32
        with self.store.db() as db:
            db.execute('''INSERT INTO contributions
                (id,owner,upload_id,title,description,visibility,status,settings,vector,created,updated)
                VALUES (?,?,?,?,?,'public','ready','{}',?,?,?)''',
                (needle, 'test-owner', '0' * 32, 'Purple Needle', 'An unrelated caption', json.dumps([-1.,0.,0.]), time.time(), time.time()))
        self.store.publish_vector(self.store.get(needle), [-1.,0.,0.])
        hits = self.store.search('Purple Needle', limit=12, vector=[1.,0.,0.])
        self.assertIn(needle, [item['id'] for item in hits])
        self.assertNotEqual(hits[0]['id'], needle)
        self.assertEqual(next(item['score'] for item in hits if item['id'] == needle), -1)
        hidden = self.store.search('Purple Needle', limit=1, vector=[1.,0.,0.], hidden=[needle])
        self.assertNotIn(needle, [row['id'] for row in hidden])

    def page(self, **changes):
        body = dict(query='a happy person dancing', model='local', limit=2, offset=0)
        body.update(changes)
        response = self.client.post('/api/tools/feedback', json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_search_pages_merge_sources_without_duplicates_and_report_exhaustion(self):
        first = self.page()
        self.assertEqual(len(first['results']), 2)
        self.assertTrue(first['has_more'])
        self.assertEqual(first['next_offset'], 2)
        second = self.page(offset=first['next_offset'])
        self.assertEqual(len(second['results']), 2)
        self.assertFalse(second['has_more'])
        rows = first['results'] + second['results']
        self.assertEqual(len({row['id'] for row in rows}), 4)
        self.assertIn(self.public_id, [row['id'] for row in rows])
        self.assertNotIn(self.private_id, [row['id'] for row in rows])
        weighted = [row['score'] * (1.02 if row['id'].startswith('contrib_') else 1) for row in rows]
        self.assertEqual(weighted, sorted(weighted, reverse=True))
        end = self.page(offset=second['next_offset'])
        self.assertEqual(end['results'], [])
        self.assertFalse(end['has_more'])
        self.assertEqual(end['next_offset'], 4)

    def test_search_pages_continue_past_sixty_and_stop_before_zero(self):
        ids = [f'positive-{i:03}' for i in range(95)] + ['zero', 'negative']
        self.base.add(ids=ids,
                      embeddings=[normalize([1., (i + 1) / 50., 0.]).tolist() for i in range(95)] + [[0.,1.,0.], [-1.,0.,0.]],
                      documents=['a happy person dancing'] * len(ids),
                      metadatas=[dict(url=f'https://example.com/{key}.gif', title=key, category='general') for key in ids])
        with patch('web.tools.local_vector', return_value=(1., 0., 0.)):
            rows, offset = [], 0
            for _ in range(20):
                data = self.page(limit=9, offset=offset)
                self.assertTrue(all(row['score'] > 0 for row in data['results']))
                rows.extend(data['results'])
                self.assertEqual(data['next_offset'], offset + len(data['results']))
                offset = data['next_offset']
                if not data['has_more']:
                    break
            else:
                self.fail('Search pagination did not reach the end')
            self.assertEqual(len(rows), 98)  # 95 new + two positive catalog + public contribution
            self.assertEqual(len({row['id'] for row in rows}), len(rows))
            self.assertTrue(set(ids[:95]) <= {row['id'] for row in rows})
            self.assertNotIn('zero', [row['id'] for row in rows])
            self.assertNotIn('negative', [row['id'] for row in rows])
            weighted = [row['score'] * (1.02 if row['id'].startswith('contrib_') else 1) for row in rows]
            self.assertEqual(weighted, sorted(weighted, reverse=True))
            self.assertEqual(self.page(offset=offset)['results'], [])

    def test_search_pages_keep_feedback_hidden_memory_and_category(self):
        original = self.page()
        memory = [dict(gif_id=self.public_id, query='a happy person dancing', query_vector=original['vector'])]
        first = self.page(limit=1, memory=memory, liked=['cat'])
        second = self.page(limit=1, offset=1, memory=memory, liked=['cat'])
        self.assertNotIn(self.public_id, [row['id'] for row in first['results'] + second['results']])
        self.assertEqual(first['hidden'], 1)
        restored = self.page(limit=60, memory=memory, liked=['cat'], unhide=True)
        self.assertIn(self.public_id, [row['id'] for row in restored['results']])
        self.assertEqual(restored['memory'], [])
        self.base.update(ids=['cat'], metadatas=[dict(category='animals')])
        filtered = self.page(category='animals')
        self.assertEqual([row['id'] for row in filtered['results']], ['cat'])
        self.assertFalse(filtered['has_more'])
        self.assertEqual(self.page(category='animals', offset=1)['results'], [])

    def test_unusable_sources_do_not_hide_later_positive_results(self):
        ids = [f'unusable-{i}' for i in range(7)]
        self.base.add(ids=ids, embeddings=[[1.,0.,0.]] * len(ids), documents=['bad source'] * len(ids),
                      metadatas=[dict(url='javascript:alert(1)', category='general')] * len(ids))
        with patch('web.tools.local_vector', return_value=(1., 0., 0.)):
            page = self.page(limit=2)
            self.assertEqual(len(page['results']), 2)
            self.assertTrue(page['has_more'])
            self.assertFalse(set(ids) & {row['id'] for row in page['results']})

    def test_search_page_offset_validation(self):
        response = self.client.post('/api/tools/feedback', json=dict(query='happy', offset=-1))
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
