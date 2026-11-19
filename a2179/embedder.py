"""Optional local sentence embedder (sentence-transformers), with clear failure reasons.

Weights are downloaded into the user's Hugging Face cache (HF_HOME), never into this folder.
Any failure is classified into one stage, reported once (log line + console line with the fix)
and the monitor falls back to regex + TF-IDF, unless the user demanded the embedder.
"""
import importlib
import importlib.metadata as md
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import prototypes
from .obs import ev
from .similarity import TfidfSim

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

STAGES = {
    "not_installed": "не установлен",
    "import_failed": "не импортируется",
    "wrong_model": "неверное имя модели",
    "offline": "нет модели в кэше, а скачивание выключено",
    "download_failed": "не скачалась модель",
    "cache_permission": "нет прав на папку кэша",
    "load_failed": "не загрузилась модель",
    "encode_failed": "ошибка при кодировании",
}

FIX_INSTALL = ("pip install -r requirements-embed.txt (torch для CPU: "
               "pip install torch --index-url https://download.pytorch.org/whl/cpu), "
               "подробности в README, раздел «Эмбеддер не работает»")


class EmbedderUnavailable(RuntimeError):
    def __init__(self, stage: str, cause: str, fix: str):
        self.stage, self.cause, self.fix = stage, cause, fix
        super().__init__(f"{stage}: {cause}")

    def traceback_text(self) -> str:
        c = self.__cause__ or self
        return "".join(traceback.format_exception(type(c), c, c.__traceback__)).strip()

    def console(self) -> str:
        return (f"[embedder] выключен, {STAGES.get(self.stage, self.stage)} ({self.stage}): {self.cause}\n"
                f"[embedder] как починить: {self.fix}\n"
                f"[embedder] работаю дальше на regex + TF-IDF")


def _short(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).strip().splitlines()[0][:240] if str(e).strip() else ''}"


def _mro_names(e: BaseException):
    return {c.__name__ for c in type(e).__mro__}


def _numpy_abi(msg: str) -> bool:
    m = msg.lower()
    return ("numpy is not available" in m or "compiled using numpy 1.x" in m
            or "_array_api not found" in m or "numpy.core.multiarray failed" in m)


def classify_import_error(e: BaseException) -> EmbedderUnavailable:
    msg = str(e)
    if _numpy_abi(msg):
        return EmbedderUnavailable("import_failed", _short(e),
                                   "torch собран под NumPy 1.x: pip install \"numpy<2\" "
                                   "(или обнови torch до версии, собранной под NumPy 2)")
    if isinstance(e, ImportError):  # includes ModuleNotFoundError
        missing = getattr(e, "name", None) or "sentence_transformers"
        return EmbedderUnavailable("not_installed", _short(e), f"нет модуля {missing}: {FIX_INSTALL}")
    if isinstance(e, OSError):  # e.g. Windows DLL load failure in torch
        return EmbedderUnavailable("import_failed", _short(e),
                                   "torch не грузит свои библиотеки. Windows: поставь Microsoft Visual C++ "
                                   "Redistributable x64 и переустанови CPU torch. Проверь, что Python 64 bit")
    return EmbedderUnavailable("import_failed", _short(e), "переустанови: " + FIX_INSTALL)


def classify_load_error(e: BaseException, model: str) -> EmbedderUnavailable:
    names = _mro_names(e)
    msg = str(e)
    low = msg.lower()
    if names & {"RepositoryNotFoundError", "GatedRepoError", "RevisionNotFoundError"} or \
            "is not a valid model identifier" in low or "repository not found" in low:
        return EmbedderUnavailable("wrong_model", _short(e),
                                   f"проверь embedder.model в config.json (сейчас {model}), "
                                   f"по умолчанию {DEFAULT_MODEL}")
    if isinstance(e, PermissionError) or "permission denied" in low:
        return EmbedderUnavailable("cache_permission", _short(e),
                                   "задай HF_HOME на папку с правами записи, например "
                                   "export HF_HOME=~/hf-cache (Windows: set HF_HOME=C:\\hf-cache)")
    if names & {"LocalEntryNotFoundError", "OfflineModeIsEnabled"} or "offline" in low or \
            "local_files_only" in low or "cannot find the requested files in the" in low:
        return EmbedderUnavailable("offline", _short(e),
                                   "один раз запусти с интернетом (без HF_HUB_OFFLINE=1 и с "
                                   "embedder.local_only=false), модель ляжет в кэш HF")
    if names & {"ConnectionError", "Timeout", "ProxyError", "SSLError", "HfHubHTTPError", "HTTPError",
                "ReadTimeout", "ConnectTimeout", "MaxRetryError"} or any(
            s in low for s in ("couldn't connect", "connection", "timed out", "proxy", "429",
                               "too many requests", "ssl", "certificate", "name resolution")):
        return EmbedderUnavailable("download_failed", _short(e),
                                   "нужен интернет до huggingface.co на первый запуск (~470 МБ). Прокси: "
                                   "HTTPS_PROXY=http://host:port. Лимит 429: подожди или задай HF_TOKEN. "
                                   "Потом можно работать офлайн")
    if isinstance(e, ImportError) or _numpy_abi(msg):
        return classify_import_error(e)
    if isinstance(e, TypeError) and "local_files_only" in msg:
        return EmbedderUnavailable("load_failed", _short(e),
                                   "старая sentence-transformers: pip install -U \"sentence-transformers>=3,<6\"")
    return EmbedderUnavailable("load_failed", _short(e),
                               "обнови пакеты: pip install -U -r requirements-embed.txt, "
                               "пришли вывод python monitor.py --check-embedder")


def classify_encode_error(e: BaseException) -> EmbedderUnavailable:
    if _numpy_abi(str(e)):
        return EmbedderUnavailable("encode_failed", _short(e),
                                   "torch и NumPy несовместимы: pip install \"numpy<2\" или новее torch")
    return EmbedderUnavailable("encode_failed", _short(e),
                               "обнови пакеты: pip install -U -r requirements-embed.txt, "
                               "пришли вывод python monitor.py --check-embedder")


def _to_matrix(out, n: int) -> np.ndarray:
    """Accept ndarray, list of lists, list of tensors or a tensor. Returns float32 (n, d), L2-normalized."""
    if hasattr(out, "detach"):  # torch tensor
        out = out.detach().cpu().numpy()
    if isinstance(out, (list, tuple)) and out and hasattr(out[0], "detach"):
        out = [t.detach().cpu().numpy() for t in out]
    M = np.asarray(out, dtype=np.float32)
    if M.ndim == 1 and n == 1:
        M = M[None, :]
    if M.ndim != 2 or M.shape[0] != n or M.shape[1] == 0:
        raise ValueError(f"unexpected embedding shape {M.shape} for {n} texts")
    if not np.all(np.isfinite(M)):
        raise ValueError("embedding contains NaN/inf")
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


class EmbedSim:
    name = "embed"

    def __init__(self, model_name: str = DEFAULT_MODEL, local_only: bool = False, pos=None, neg=None):
        try:
            st = importlib.import_module("sentence_transformers")
            ST = st.SentenceTransformer
        except Exception as e:
            raise classify_import_error(e) from e
        self.model_name = model_name
        t0 = time.monotonic()
        try:
            kw = {"device": "cpu"}
            if local_only:
                kw["local_files_only"] = True
            try:
                self.model = ST(model_name, **kw)
            except TypeError as e:
                if "local_files_only" not in str(e):
                    raise
                os.environ["HF_HUB_OFFLINE"] = "1"  # old versions: same effect via env
                kw.pop("local_files_only")
                self.model = ST(model_name, **kw)
        except EmbedderUnavailable:
            raise
        except Exception as e:
            raise classify_load_error(e, model_name) from e
        self.load_s = time.monotonic() - t0
        self._kw = {"normalize_embeddings": True, "convert_to_numpy": True, "show_progress_bar": False,
                    "batch_size": 32}
        self.pos = list(pos or prototypes.POSITIVE)
        self.neg = list(neg or prototypes.NEGATIVE)
        self.P = self._enc(self.pos)
        self.N = self._enc(self.neg)
        if self.P.shape[1] != self.N.shape[1]:
            raise EmbedderUnavailable("encode_failed", "prototype dims differ", "переустанови модель")

    def _enc(self, texts) -> np.ndarray:
        texts = list(texts)
        try:
            try:
                out = self.model.encode(texts, **self._kw)
            except TypeError:
                # very old encode() signature: drop optional kwargs one by one
                for k in ("normalize_embeddings", "show_progress_bar", "batch_size"):
                    self._kw.pop(k, None)
                    try:
                        out = self.model.encode(texts, **self._kw)
                        break
                    except TypeError:
                        continue
                else:
                    out = self.model.encode(texts)
            return _to_matrix(out, len(texts))
        except EmbedderUnavailable:
            raise
        except Exception as e:
            raise classify_encode_error(e) from e

    def score(self, text: str):
        v = self._enc([text])[0]
        return float(np.max(self.P @ v)), float(np.max(self.N @ v))

    def score_many(self, texts):
        V = self._enc(texts)
        return list(zip((V @ self.P.T).max(axis=1).tolist(), (V @ self.N.T).max(axis=1).tolist()))


def embedder_wanted(cfg: dict) -> str:
    """'off' | 'auto' | 'required'."""
    e = cfg.get("embedder", {})
    v = str(e.get("enabled", "auto")).strip().lower()
    if v in ("false", "0", "no", "off", "none"):
        return "off"
    if v in ("required", "require") or e.get("required") is True:
        return "required"
    return "auto"


def load_similarity(cfg: dict, log=None, require: bool = False, console=sys.stderr, verbose: bool = False):
    """Returns (sim, mode, error). mode is 'embed' or 'tfidf'.
    Raises EmbedderUnavailable only if the embedder is required (CLI flag or config)."""
    want = "required" if require else embedder_wanted(cfg)
    if want == "off":
        return TfidfSim(), "tfidf", None
    e = cfg.get("embedder", {})
    try:
        sim = EmbedSim(e.get("model") or DEFAULT_MODEL, bool(e.get("local_only", False)))
        if log:
            ev(log, "embedder_ready", model=sim.model_name, load_s=round(sim.load_s, 2))
        return sim, "embed", None
    except EmbedderUnavailable as err:
        if log:
            ev(log, "embedder_unavailable", logging.WARNING, stage=err.stage, cause=err.cause, fix=err.fix,
               required=want == "required", traceback=err.traceback_text()[-4000:])
        if verbose and console:
            print(err.traceback_text(), file=console)
        if want == "required":
            raise
        if console:
            print(err.console(), file=console)
        return TfidfSim(), "tfidf", err


# ---------------------------------------------------------------- diagnostics

def _ver(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"
    except Exception as e:  # pragma: no cover
        return f"? ({type(e).__name__})"


def hf_cache_dir() -> Path:
    if os.environ.get("SENTENCE_TRANSFORMERS_HOME"):
        return Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]).expanduser()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "huggingface")
    return Path(home).expanduser() / "hub"


def check_embedder(cfg: dict, out=sys.stdout) -> int:
    """Print a diagnostic report. Returns 0 if the embedder works, 3 if not."""
    p = lambda *a: print(*a, file=out)  # noqa: E731
    e = cfg.get("embedder", {})
    model = e.get("model") or DEFAULT_MODEL
    cache = hf_cache_dir()
    p("== A2179 embedder check ==")
    p(f"python:     {sys.version.split()[0]} ({platform.python_implementation()}, {8 * np.dtype('intp').itemsize} bit)")
    p(f"os:         {platform.system()} {platform.release()} {platform.machine()}")
    for pkg in ("torch", "sentence-transformers", "transformers", "huggingface-hub", "numpy", "tokenizers"):
        p(f"{pkg + ':':<23} {_ver(pkg)}")
    p(f"model:      {model}  (local_only={bool(e.get('local_only', False))})")
    p(f"config:     embedder.enabled={e.get('enabled', 'auto')}  -> {embedder_wanted(cfg)}")
    writable = os.access(cache if cache.exists() else cache.parent if cache.parent.exists() else Path.home(), os.W_OK)
    p(f"cache dir:  {cache}  (exists={cache.exists()}, writable={writable})")
    for k in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HTTPS_PROXY", "HTTP_PROXY"):
        if os.environ.get(k):
            p(f"env:        {k}={'set' if 'PROXY' in k else os.environ[k]}")
    t0 = time.monotonic()
    try:
        sim = EmbedSim(model, bool(e.get("local_only", False)))
    except EmbedderUnavailable as err:
        p(f"load:       FAILED after {time.monotonic() - t0:.1f} s")
        p("---- traceback ----")
        p(err.traceback_text())
        p("-------------------")
        p(err.console())
        p("mode:       tfidf (fallback)")
        return 3
    p(f"load:       ok in {sim.load_s:.1f} s (first run includes the ~470 MB download)")
    try:
        t1 = time.monotonic()
        a = sim.score("LCD ekran za MacBook Air A2179 2020")
        b = sim.score("Futrola za MacBook Air A2179")
        dim = sim.P.shape[1]
        p(f"encode:     ok, dim={dim}, dtype={sim.P.dtype}, {1000 * (time.monotonic() - t1) / 2:.0f} ms/text")
        p(f"sanity:     screen pos/neg={a[0]:.2f}/{a[1]:.2f}  case pos/neg={b[0]:.2f}/{b[1]:.2f}  "
          f"({'ok' if a[0] - a[1] > b[0] - b[1] else 'odd: screen should score above case'})")
    except EmbedderUnavailable as err:
        p(err.traceback_text())
        p(err.console())
        p("mode:       tfidf (fallback)")
        return 3
    th = cfg.get("thresholds", {}).get("embed", {})
    p(f"thresholds: embed pos={th.get('pos')} margin={th.get('margin')} "
      f"({'calibrated' if th.get('calibrated') else 'NOT calibrated: run python eval/run_eval.py --write-thresholds'})")
    p("mode:       embed")
    return 0
