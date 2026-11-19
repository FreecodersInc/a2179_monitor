"""Similarity channel: char n-gram TF-IDF (always available). The optional embedder is in embedder.py.

score(text) -> (sim_pos, sim_neg): max cosine to positive and negative prototypes.
"""
import math
from collections import Counter

from .normalize import normalize
from . import prototypes


def char_ngrams(text: str, nmin: int = 3, nmax: int = 5):
    """char_wb-style n-grams over normalized words, padded with spaces."""
    out = []
    for w in normalize(text).split():
        w = f" {w} "
        for n in range(nmin, nmax + 1):
            if len(w) < n:
                continue
            out.extend(w[i:i + n] for i in range(len(w) - n + 1))
    return out


class TfidfSim:
    name = "tfidf"

    def __init__(self, pos=None, neg=None):
        self.pos = list(pos or prototypes.POSITIVE)
        self.neg = list(neg or prototypes.NEGATIVE)
        docs = [Counter(char_ngrams(d)) for d in self.pos + self.neg]
        df = Counter()
        for d in docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1.0 for g, c in df.items()}
        self.default_idf = math.log((1 + n) / 1) + 1.0
        self.P = [self._vec(c) for c in docs[:len(self.pos)]]
        self.N = [self._vec(c) for c in docs[len(self.pos):]]

    def _vec(self, counts: Counter) -> dict:
        v = {g: (1 + math.log(c)) * self.idf.get(g, self.default_idf) for g, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    @staticmethod
    def _cos(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(g, 0.0) for g, x in a.items())

    def score(self, text: str):
        v = self._vec(Counter(char_ngrams(text)))
        return max(self._cos(v, p) for p in self.P), max(self._cos(v, q) for q in self.N)
