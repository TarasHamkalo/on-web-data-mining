from dataclasses import dataclass, field
from spacy.tokens import Doc

@dataclass
class Article:
    title: str
    content: str
    category: str

    document: Doc = field(default=None, repr=False, init=False)
