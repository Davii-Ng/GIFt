"""Shared metadata rules and community search ranking."""
import re
import unicodedata

import numpy as np

TEXT_LIMIT = 1000
VECTOR_VERSION = 2
COMMUNITY_WEIGHT = 1.02


def normalize_details(title='', description=''):
    title, description = title.strip(), description.strip()
    if not title and not description:
        raise ValueError('Enter a name or a description. You can leave one field empty, but not both.')
    return title or description, description or title


def tokens(text):
    text = ''.join(char for char in unicodedata.normalize('NFKD', text.casefold())
                   if not unicodedata.combining(char))
    return re.findall(r'[^\W_]+', text, flags=re.UNICODE)


def query_parts(query):
    phrase = len(query) > 1 and query[0] in '\"“' and query[-1] in '\"”'
    words = tokens(query)
    return words, phrase


def lexical_priority(query, title, description):
    words, phrase = query_parts(query)
    if not words:
        return 0
    normalized = ' '.join(words)
    fields = [' '.join(tokens(value)) for value in (title, description)]
    if normalized in fields:
        return 3
    if phrase:
        return 2 if any(f' {normalized} ' in f' {value} ' for value in fields) else 0
    available = set(tokens(title)) | set(tokens(description))
    return 2 if set(words) <= available else 0


def metadata_vector(model, title, description):
    """Weight the name and description separately so neither field is truncated away."""
    title_vector = np.asarray(model.encode(title, normalize_embeddings=True), dtype=np.float32)
    if title == description:
        vector = title_vector
    else:
        description_vector = np.asarray(model.encode(description, normalize_embeddings=True), dtype=np.float32)
        vector = 0.6 * title_vector + 0.4 * description_vector
    length = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.isfinite(vector).all() or length <= 0:
        raise ValueError('Invalid metadata embedding')
    return (vector / length).tolist()


def community_rank(item):
    score = item.get('score', -1)
    # Apply a single 2% preference to positive relevance, without keyword bonuses.
    # Zero/negative similarity cannot earn a boost; reported scores stay unchanged.
    weighted = score * COMMUNITY_WEIGHT if score > 0 else score
    return (weighted, score)


def merge_results(items, extra, limit):
    community = {item['id']: item for item in extra}
    by_id = {item['id']: item for item in [*items, *extra]}

    def key(item):
        if item['id'] in community:
            return (*community_rank(item), item['id'])
        return (item.get('score', -1), item.get('score', -1), item['id'])

    # Keep cosine similarity intact; preference is a separate ranking decision.
    return sorted(by_id.values(), key=key, reverse=True)[:limit]
