"""Fake sentence_transformers modules to exercise the embedder path without torch or weights."""
import contextlib
import sys
import types
import zlib

import numpy as np

from a2179.normalize import normalize

DIM = 64


def _vec(text):
    """Deterministic bag of char trigrams hashed into DIM dims (stands in for a real model)."""
    v = np.zeros(DIM, dtype=np.float64)
    t = f" {normalize(text)} "
    for i in range(len(t) - 2):
        v[zlib.crc32(t[i:i + 3].encode()) % DIM] += 1.0
    return v


class GoodST:
    def __init__(self, name, device=None, local_files_only=False, **kw):
        self.name = name

    def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=None,
               batch_size=32):
        M = np.stack([_vec(t) for t in texts]).astype(np.float32)
        if normalize_embeddings:
            M /= np.linalg.norm(M, axis=1, keepdims=True)
        return M


class OldApiST(GoodST):
    """No local_files_only in __init__, no normalize_embeddings in encode, returns a list of lists."""
    def __init__(self, name, device=None):
        self.name = name

    def encode(self, texts, batch_size=32, show_progress_bar=None, convert_to_numpy=True):
        return [list(_vec(t)) for t in texts]


class FakeTensor:
    def __init__(self, a):
        self.a = a

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class TensorST(GoodST):
    def encode(self, texts, **kw):
        return FakeTensor(np.stack([_vec(t) for t in texts]).astype(np.float32))


class BadShapeST(GoodST):
    def encode(self, texts, **kw):
        return np.zeros(len(texts), dtype=np.float32)  # 1-D: wrong


class NumpyAbiST(GoodST):
    def encode(self, texts, **kw):
        raise RuntimeError("Numpy is not available")


class BadSignatureST(GoodST):
    def encode(self, texts, *, flavour):  # nothing we pass is accepted
        raise AssertionError("unreachable")


def raising(exc):
    class R(GoodST):
        def __init__(self, *a, **k):
            raise exc
    return R


# exception classes named like the huggingface_hub ones
class RepositoryNotFoundError(OSError):
    pass


class LocalEntryNotFoundError(OSError):
    pass


class HfHubHTTPError(OSError):
    pass


@contextlib.contextmanager
def fake_module(cls):
    """Install sentence_transformers with SentenceTransformer=cls. cls=None: module missing."""
    saved = sys.modules.get("sentence_transformers", "absent")
    if cls is None:
        sys.modules["sentence_transformers"] = None  # makes `import` raise ImportError
    else:
        m = types.ModuleType("sentence_transformers")
        m.SentenceTransformer = cls
        sys.modules["sentence_transformers"] = m
    try:
        yield
    finally:
        if saved == "absent":
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = saved


class _RaisingFinder:
    def __init__(self, exc):
        self.exc = exc

    def find_spec(self, name, path=None, target=None):
        if name == "sentence_transformers":
            raise self.exc
        return None


@contextlib.contextmanager
def import_raises(exc):
    """`import sentence_transformers` raises exc (e.g. OSError from a torch DLL on Windows)."""
    saved = sys.modules.pop("sentence_transformers", "absent")
    f = _RaisingFinder(exc)
    sys.meta_path.insert(0, f)
    try:
        yield
    finally:
        sys.meta_path.remove(f)
        if saved != "absent":
            sys.modules["sentence_transformers"] = saved
