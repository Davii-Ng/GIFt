import json
import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import numpy as np

from web.engine import ROOT, Retriever, close_database, document_for, get_collection, get_model, read_catalog
from web.ingest import ingest
from web.prepare_tgif import load_tgif
from web.server import app

SAMPLE_PATH = ROOT / "data" / "gifs.json"


class CatalogTests(unittest.TestCase):
    def test_catalog_rejects_duplicate_ids_and_unsafe_urls(self):
        rows = read_catalog(SAMPLE_PATH)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps([rows[0], rows[0]]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_catalog(path)
            row = dict(rows[0], url="javascript:alert(1)")
            path.write_text(json.dumps([row]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                read_catalog(path)

    def test_tgif_import_cleans_and_samples_reproducibly(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "input.tsv"
            rows = [(f"https://example.com/{i}.gif", f"description {i}") for i in range(30)]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerows(rows + [rows[0], ("", "missing URL"), ("https://example.com/empty.gif", ""), ("javascript:alert(1)", "bad URL")])
            first = load_tgif(path, limit=9)
            self.assertEqual(first, load_tgif(path, limit=9))
            expected = np.random.RandomState(42).choice(30, size=9, replace=False)
            self.assertEqual([row["description"] for row in first], [f"description {i}" for i in expected])
            self.assertEqual(len(load_tgif(path, limit=100)), 30)
            self.assertTrue(all(row["category"] == "general" and row["source"] == "TGIF" for row in first))
            self.assertEqual(len({row["id"] for row in first}), 9)
            with self.assertRaises(ValueError):
                load_tgif(path, limit=0)

    def test_tgif_preserves_every_caption_for_duplicate_urls(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "captions.tsv"
            path.write_text("https://example.com/a.gif\ta dog runs\n"
                            "https://example.com/a.gif\ta puppy plays outside\n"
                            "https://example.com/b.gif\ta cat sleeps\n", encoding="utf-8")
            rows = load_tgif(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(row["caption_count"] for row in rows), 3)
            dog = next(row for row in rows if row["url"].endswith("/a.gif"))
            self.assertIn("a dog runs", dog["description"])
            self.assertIn("a puppy plays outside", dog["description"])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use a separate real database so tests never contend with the running app.
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT)
        cls.addClassCleanup(cls.directory.cleanup)
        cls.db_path = Path(cls.directory.name) / "chroma"
        cls.addClassCleanup(lambda: close_database(cls.db_path))
        cls.records = read_catalog(SAMPLE_PATH)
        cls.contribution_patch = patch.dict(os.environ, {'CONTRIBUTION_DATA_DIR': str(Path(cls.directory.name) / 'contributions')})
        cls.contribution_patch.start()
        cls.addClassCleanup(cls.contribution_patch.stop)
        ingest(path=SAMPLE_PATH, db_path=cls.db_path)
        cls.catalog_patch = patch("web.server.read_catalog", return_value=cls.records)
        cls.catalog_patch.start()
        cls.addClassCleanup(cls.catalog_patch.stop)
        cls.retriever_patch = patch("web.server.Retriever", side_effect=lambda: Retriever(path=cls.db_path))
        cls.retriever_patch.start()
        cls.addClassCleanup(cls.retriever_patch.stop)
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_api_and_private_files(self):
        self.assertTrue(self.client.get("/api/health").json()["ready"])
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/app.js").status_code, 200)
        self.assertEqual(self.client.get("/api-client.js").status_code, 200)
        self.assertEqual(self.client.get('/gif-actions.js').status_code, 200)
        self.assertEqual(self.client.get('/system-log.js').status_code, 200)
        for path in ("/.chroma/chroma.sqlite3", "/server.py", "/.venv/pyvenv.cfg"):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_live_server_cors(self):
        response = self.client.get("/api/catalog", headers={"Origin": "http://127.0.0.1:5500"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://127.0.0.1:5500")
        self.assertIn("application/json", response.headers["content-type"])
        response = self.client.get("/api/catalog", headers={"Origin": "https://unrelated.example"})
        self.assertNotIn("access-control-allow-origin", response.headers)
        response = self.client.options('/api/contributions', headers={
            'Origin': 'http://127.0.0.1:5500', 'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'authorization,content-type'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('authorization', response.headers.get('access-control-allow-headers', '').lower())
        response = self.client.options('/api/contributions/test/visibility', headers={
            'Origin': 'http://127.0.0.1:5500', 'Access-Control-Request-Method': 'PATCH',
            'Access-Control-Request-Headers': 'authorization,content-type'})
        self.assertEqual(response.status_code, 200)

    def test_upload_body_limit_and_missing_auth(self):
        response = self.client.post('/api/contributions/uploads', content=b'not a video')
        self.assertEqual(response.status_code, 401)
        response = self.client.post('/api/contributions/uploads', content=b'x', headers={
            'Authorization': 'Bearer invalid', 'Content-Length': str(53 * 1024 * 1024)})
        self.assertEqual(response.status_code, 413)

    def test_semantic_ranking_and_category(self):
        response = self.client.get("/api/search", params={"q": "I desperately want to go to bed", "limit": 3})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["results"][0]["id"], "sleepy")
        self.assertEqual(data["count"], 3)
        scores = [row["similarity"] for row in data["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        response = self.client.get("/api/search", params={"q": "cute kitten", "category": "animals", "limit": 60})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["results"])
        expected = sum(row["category"] == "animals" for row in self.records)
        self.assertEqual(response.json()["count"], expected)
        self.assertTrue(all(row["category"] == "animals" for row in response.json()["results"]))

    def test_invalid_query(self):
        for params in ({"q": " "}, {"q": "a" * 301}, {"q": "cat", "limit": 0}, {"q": "cat", "category": "unknown"}):
            self.assertEqual(self.client.get("/api/search", params=params).status_code, 422)

    def test_paged_catalog_and_saved_lookup(self):
        first = self.client.get("/api/catalog", params={"limit": 8}).json()
        second = self.client.get("/api/catalog", params={"limit": 8, "offset": 8}).json()
        self.assertEqual(len(first["gifs"]), 8)
        self.assertEqual(first["total"], len(self.records))
        self.assertTrue(first["has_more"])
        self.assertFalse({row["id"] for row in first["gifs"]} & {row["id"] for row in second["gifs"]})
        saved_id = self.records[-1]["id"]
        saved = self.client.get("/api/catalog", params=[("ids", saved_id), ("ids", "not-found")]).json()
        self.assertEqual([row["id"] for row in saved["gifs"]], [saved_id])
        for params in ({"limit": 0}, {"limit": 61}, {"offset": -1}):
            self.assertEqual(self.client.get("/api/catalog", params=params).status_code, 422)

    def test_shuffled_catalog_is_stable_across_pages(self):
        def ids(**params):
            response = self.client.get("/api/catalog", params=params)
            self.assertEqual(response.status_code, 200)
            return [row["id"] for row in response.json()["gifs"]]

        first = ids(seed=7, limit=8)
        second = ids(seed=7, limit=8, offset=8)
        self.assertEqual(first + second, ids(seed=7, limit=16))
        self.assertEqual(len(set(first + second)), 16)
        self.assertNotEqual(first, ids(seed=8, limit=8))
        self.assertEqual(ids(limit=8), [row["id"] for row in self.records[:8]])

    def test_chroma_matches_notebook_cosine(self):
        query = "a cute kitten"
        model = get_model()
        documents = [document_for(row) for row in self.records]
        self.assertEqual(documents, [row["description"] for row in self.records])
        stored = model.encode(documents, normalize_embeddings=True)
        vector = model.encode(query, normalize_embeddings=True)
        self.assertEqual(vector.shape, (384,))
        scores = stored @ vector
        best = np.argsort(-scores)[:5]
        results = self.client.get("/api/search", params={"q": query, "limit": 5}).json()["results"]
        self.assertEqual([row["id"] for row in results], [self.records[index]["id"] for index in best])
        for row, index in zip(results, best):
            self.assertAlmostEqual(row["score"], float(scores[index]), places=3)
            self.assertEqual(row["similarity"], row["score"])

    def test_ingestion_is_idempotent(self):
        before = get_collection(path=self.db_path).count()
        report = ingest(path=SAMPLE_PATH, db_path=self.db_path)
        self.assertEqual(report["indexed"], 0)
        self.assertEqual(report["removed"], 0)
        self.assertEqual(report["total"], before)


if __name__ == "__main__":
    unittest.main()
