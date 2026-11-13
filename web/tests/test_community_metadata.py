import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from web.contribution_search import VECTOR_VERSION, community_rank, merge_results, metadata_vector
from web.contributions import ContributionService, CreateGIF, UpdateGIF


class NamedModel:
    def encode(self, text, **kwargs):
        return np.array([1., 0., 0.]) if 'name' in text else np.array([0., 1., 0.])


class CommunityMetadataTests(unittest.TestCase):
    def test_metadata_normalization_is_identical_for_create_and_edit(self):
        for fields, expected in [({'title': ' Name ', 'description': ''}, ('Name', 'Name')),
                                 ({'description': ' Long description ' * 12}, (('Long description ' * 12).strip(), ('Long description ' * 12).strip())),
                                 ({'title': 'X'}, ('X','X')),
                                 ({'title': 'Name', 'description': 'Description'}, ('Name','Description'))]:
            # strip() removes only the ends, preserving every internal character.
            if 'description' in fields and fields['description'].strip() and not fields.get('title'):
                expected = (fields['description'].strip(), fields['description'].strip())
            for cls, additional in [(CreateGIF, dict(upload_id='0' * 32, end=1)), (UpdateGIF, dict(visibility='public'))]:
                with self.subTest(fields=fields, model=cls.__name__):
                    result = cls(**fields, **additional)
                    self.assertEqual((result.title, result.description), expected)
        for cls, additional in [(CreateGIF, dict(upload_id='0' * 32, end=1)), (UpdateGIF, dict(visibility='private'))]:
            for fields in ({}, {'title': ' ', 'description': '\n'}, {'title': 'x' * 1001}):
                with self.assertRaises(ValueError):
                    cls(**fields, **additional)

    def test_vector_contains_both_fields_and_title_edits_change_it(self):
        vector = metadata_vector(NamedModel(), 'name', 'description')
        self.assertGreater(vector[0], 0)
        self.assertGreater(vector[1], 0)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=6)
        self.assertNotEqual(vector, metadata_vector(NamedModel(), 'description', 'description'))

    def test_community_priority_does_not_change_reported_cosine_or_promote_unrelated_gifs(self):
        base = {'id': 'catalog', 'score': .95}
        exact = {'id': 'community', 'score': .2, 'match_priority': 3}
        self.assertEqual(merge_results([base], [exact], 2)[0], base)
        self.assertEqual(exact['score'], .2)
        unrelated = {'id': 'unrelated', 'score': .1}
        self.assertEqual(merge_results([base], [unrelated], 2)[0], base)
        close = {'id': 'close', 'score': .94}
        self.assertEqual(merge_results([base], [close], 2)[0], close)

    def test_single_two_percent_weight_has_no_stacked_keyword_bonus(self):
        for score in (.1, .2, .5, .9, 1.):
            for priority in (0, 2, 3):
                with self.subTest(score=score, priority=priority):
                    item = dict(id='community', score=score, similarity=score, match_priority=priority)
                    original = dict(item)
                    self.assertAlmostEqual(community_rank(item)[0], score * 1.02)
                    self.assertEqual(item, original)
        for score in (-1., -.1, 0.):
            self.assertEqual(community_rank(dict(score=score, match_priority=3))[0], score)

    def test_relative_weight_prefers_close_matches_but_not_weaker_exact_names(self):
        catalog = {'id': 'catalog', 'score': .6}
        close = {'id': 'community', 'score': .59, 'match_priority': 0}
        self.assertEqual(merge_results([catalog], [close], 2)[0], close)  # .6018 > .6
        weaker_exact = dict(close, score=.58, match_priority=3)
        self.assertEqual(merge_results([catalog], [weaker_exact], 2)[0], catalog)  # .5916 < .6
        # Merging a candidate retrieved through both paths must not multiply twice.
        self.assertEqual(merge_results([catalog, weaker_exact], [weaker_exact], 2), [catalog, weaker_exact])
        negative = {'id': 'community', 'score': -.1, 'match_priority': 3}
        self.assertEqual(merge_results([{'id': 'catalog', 'score': -.09}], [negative], 2)[0]['id'], 'catalog')

    def test_legacy_vectors_are_upgraded_without_changing_metadata_or_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # First create the existing schema, then mark its vector as the legacy version.
            service = ContributionService(root, model_factory=NamedModel)
            key = 'contrib_' + 'd' * 32
            try:
                with service.db() as db:
                    db.execute('''INSERT INTO contributions
                        (id,owner,upload_id,title,description,visibility,status,settings,vector,created,updated)
                        VALUES (?,?,?,?,?,'public','ready','{}',?,1,2)''',
                        (key, 'owner', '0' * 32, 'name', 'description', json.dumps([0.,1.,0.])))
                service.file_path(key).write_bytes(b'original GIF bytes')
            finally:
                service.close()
            reopened = ContributionService(root, model_factory=NamedModel)
            try:
                row = reopened.get(key, 'owner')
                self.assertEqual(row['vector_version'], VECTOR_VERSION)
                self.assertEqual(json.loads(row['vector']), metadata_vector(NamedModel(), 'name', 'description'))
                self.assertEqual((row['title'], row['description'], row['created'], row['updated']), ('name','description',1,2))
                self.assertEqual(reopened.file_path(key).read_bytes(), b'original GIF bytes')
                self.assertEqual(reopened.search('name')[0]['id'], key)
                self.assertEqual(reopened.reindex_metadata(), 0)
                old_vector = row['vector']
                edited = reopened.edit(key, 'owner', UpdateGIF(title='description', description='description', visibility='public'))
                self.assertNotEqual(reopened.get(key)['vector'], old_vector)
                self.assertEqual(reopened.search('"name"'), [])
                reopened.edit(key, 'owner', UpdateGIF(title='private secret', visibility='private'))
                self.assertEqual(reopened.search('"private secret"'), [])
            finally:
                reopened.close()
