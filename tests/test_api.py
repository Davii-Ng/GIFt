import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app


class SearchApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_search_rejects_invalid_parameters(self):
        invalid_params = (
            {"q": ""},
            {"q": "   "},
            {"q": "x" * 501},
            {"q": "gif", "k": 0},
            {"q": "gif", "k": 21},
        )
        for params in invalid_params:
            with self.subTest(params=params):
                response = self.client.get("/search", params=params)
                self.assertEqual(response.status_code, 422)

    @patch("api.search_gif", return_value=[{"url": "https://example.com/a.gif", "score": 0.9}])
    def test_search_returns_results(self, search_gif):
        response = self.client.get("/search", params={"q": "a dog", "k": 4})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [{"url": "https://example.com/a.gif", "score": 0.9}])
        search_gif.assert_called_once_with("a dog", 4)

    @patch("api.search_gif", side_effect=RuntimeError("index missing"))
    def test_search_reports_missing_index(self, search_gif):
        response = self.client.get("/search", params={"q": "a dog"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "index missing"})
        search_gif.assert_called_once_with("a dog", 9)


if __name__ == "__main__":
    unittest.main()