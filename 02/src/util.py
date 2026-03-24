import re
import hashlib


def readable_id(sentence: str, length: int = 4) -> str:
    h = hashlib.md5(sentence.encode()).hexdigest()[:length]
    slug = re.sub(r'\W+', '_', sentence.lower()).strip('_')[:30]

    return f"{slug}_{h}"