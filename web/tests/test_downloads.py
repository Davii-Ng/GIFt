"""Downloads stay attached to known IDs and validate remote bytes and redirects."""
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.gif_downloads import MAX_DOWNLOAD_BYTES, router, validate_remote_url


GIF = b'GIF89a' + b'test image bytes'


def remote_response(status=200, headers=None, chunks=None):
    result = MagicMock()
    result.__enter__.return_value = result
    result.status_code = status
    result.headers = headers or {}
    result.iter_content.return_value = iter([GIF] if chunks is None else chunks)
    return result


class CatalogDownloadTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.state.by_id = {'known': {'id': 'known', 'title': 'A happy moment', 'url': 'https://example.com/image.gif'}}
        self.app.state.contributions = object()
        self.app.include_router(router)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.dns = patch('web.gif_downloads.socket.getaddrinfo', return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))])
        self.dns.start(); self.addCleanup(self.dns.stop)

    def test_known_gif_download_has_real_bytes_and_attachment_headers(self):
        with patch('web.gif_downloads.requests.get', return_value=remote_response()) as fetch:
            response = self.client.get('/api/gifs/known/download')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, GIF)
        self.assertEqual(response.headers['content-type'], 'image/gif')
        self.assertIn('attachment', response.headers['content-disposition'])
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(fetch.call_args.args[0], 'https://example.com/image.gif')
        self.assertFalse(fetch.call_args.kwargs['allow_redirects'])

    def test_unknown_ids_never_fetch_arbitrary_remote_urls(self):
        with patch('web.gif_downloads.requests.get') as fetch:
            self.assertEqual(self.client.get('/api/gifs/missing/download?url=https://example.org/secret').status_code, 404)
            fetch.assert_not_called()

    def test_dead_html_and_oversized_sources_have_clear_errors(self):
        cases = [(remote_response(status=404), 502),
                 (remote_response(chunks=[b'<html>not a gif</html>']), 502),
                 (remote_response(headers={'Content-Length': str(MAX_DOWNLOAD_BYTES + 1)}), 413),
                 (remote_response(chunks=[b'GIF89a' + b'x' * MAX_DOWNLOAD_BYTES]), 413)]
        for fake, expected in cases:
            with self.subTest(expected=expected), patch('web.gif_downloads.requests.get', return_value=fake):
                response = self.client.get('/api/gifs/known/download')
                self.assertEqual(response.status_code, expected)
                self.assertIn('detail', response.json())
        with patch('web.gif_downloads.requests.get', side_effect=requests.Timeout):
            self.assertEqual(self.client.get('/api/gifs/known/download').status_code, 504)

    def test_redirect_to_private_network_is_blocked(self):
        with patch('web.gif_downloads.requests.get', return_value=remote_response(status=302, headers={'Location': 'https://127.0.0.1/secret'})) as fetch:
            self.assertEqual(self.client.get('/api/gifs/known/download').status_code, 502)
            self.assertEqual(fetch.call_count, 1)
        for url in ('file:///etc/passwd', 'http://example.com/a.gif', 'https://example.com:8080/a.gif', 'https://user:pass@example.com/a.gif', 'https://192.168.1.1/a.gif'):
            with self.subTest(url=url), self.assertRaises(Exception):
                validate_remote_url(url)

    def test_https_cdn_redirects_are_supported(self):
        with patch('web.gif_downloads.requests.get', side_effect=[
            remote_response(status=302, headers={'Location': 'https://cdn.example.com/a.gif'}), remote_response()
        ]) as fetch:
            response = self.client.get('/api/gifs/known/download')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, GIF)
        self.assertEqual(fetch.call_count, 2)
