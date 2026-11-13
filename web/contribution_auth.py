"""Verify Firebase user tokens without requiring a service-account private key."""
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

import cachecontrol
import requests
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.id_token import verify_firebase_token

bearer = HTTPBearer(auto_error=False)
_verification_lock = threading.Lock()


@lru_cache(maxsize=1)
def firebase_project():
    project = os.getenv('FIREBASE_PROJECT_ID')
    if not project:
        config = Path(__file__).with_name('firebase-config.js')
        text = config.read_text(encoding='utf-8') if config.exists() else ''
        match = re.search(r'''projectId\s*:\s*['"]([a-z0-9-]+)['"]''', text)
        project = match.group(1) if match else ''
    if not project:
        raise HTTPException(503, 'Set FIREBASE_PROJECT_ID before enabling contributions.')
    return project


@lru_cache(maxsize=1)
def verification_request():
    request = GoogleRequest(session=cachecontrol.CacheControl(requests.Session()))

    def bounded_request(*args, **kwargs):
        kwargs['timeout'] = 10
        return request(*args, **kwargs)
    return bounded_request


def current_contributor(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if not credentials or credentials.scheme.lower() != 'bearer':
        raise HTTPException(401, 'Log in to create and manage your GIFs.')
    if len(credentials.credentials) > 10000:
        raise HTTPException(401, 'Invalid account token.')
    try:
        project = firebase_project()
        with _verification_lock:
            claims = verify_firebase_token(credentials.credentials, verification_request(), audience=project)
        uid = claims.get('sub', '')
        if claims.get('iss') != f'https://securetoken.google.com/{project}' or not isinstance(uid, str) or not 1 <= len(uid) <= 128:
            raise ValueError('Invalid identity')
    except HTTPException:
        raise
    except (ValueError, TypeError):
        raise HTTPException(401, 'Your session expired. Log in again.') from None
    except Exception:
        raise HTTPException(503, 'Account verification is temporarily unavailable. Please try again.') from None
    return {'uid': uid}
