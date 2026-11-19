"""Optional local sentence embedder (sentence-transformers), with clear failure reasons.

Weights are downloaded into the user's Hugging Face cache (HF_HOME), never into this folder.
Any failure is classified into one stage, reported once (log line + console line with the fix)
and the monitor falls back to regex + TF-IDF, unless the user demanded the embedder.

Native crashes (segfault in torch/CUDA, abort, hang) cannot be caught in-process, so the
embedder is first loaded in a probe child process (a2179/embed_probe.py, see run_probe):
same model, same device, load plus a test encode. Only if the child reports success does
the main process load the model itself, on the device the child used. A child that dies by
a signal, exits non-zero or runs past embedder.probe_timeout_s is reported as stage
`crashed` / `probe_timeout` / `probe_failed` and the monitor falls back like for any other
failure. Chosen design: the probe gates the in-process load, encoding during the run stays
in-process (fast, no IPC per listing). A setup that passes load + encode in the child and
still crashes later in the main process is not covered; embedder.enabled=false or
embedder.device=cpu is the switch then.

Device: embedder.device = auto | cuda | cpu (or any torch device string such as cuda:1 or
mps). auto = cuda if torch.cuda.is_available() else cpu.
Import order: torch is imported before sentence_transformers (which pulls sklearn and its
OpenMP runtime), so torch's libgomp is loaded first.
"""
import importlib
import importlib.metadata as md
import importlib.util
import json
import logging
import os
import platform
import signal
import subprocess
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
    "crashed": "процесс проверки эмбеддера упал (нативный крэш)",
    "probe_timeout": "проверка эмбеддера зависла",
    "probe_failed": "процесс проверки эмбеддера завершился с ошибкой",
}

FIX_CRASH = ("крэш внутри torch при работе с CUDA, обычно драйвер NVIDIA (в WSL2 он ставится на Windows, "
             "внутри WSL свой драйвер не ставить) не тянет CUDA сборки torch. Проверь: nvidia-smi (справа вверху "
             "максимальная CUDA драйвера), python -c \"import torch; print(torch.__version__, torch.version.cuda, "
             "torch.cuda.is_available())\" и python -c \"import torch; torch.zeros(1).cuda()\". Версия CUDA у "
             "torch (cuXXX) не должна быть новее, чем показывает nvidia-smi: обнови драйвер NVIDIA для WSL на "
             "Windows или поставь torch под CUDA постарше (pytorch.org). Пока можно embedder.device=cpu. "
             "Монитор при этом работает на regex + TF-IDF")
FIX_CRASH_CPU = ("крэш даже на cpu, хотя CUDA спрятана (CUDA_VISIBLE_DEVICES пусто). Пришли вывод "
                 "python monitor.py --check-embedder и pip freeze | grep -iE \"torch|sentence|transformers|numpy\". "
                 "Монитор при этом работает на regex + TF-IDF")


def fix_crash(device) -> str:
    return FIX_CRASH_CPU if str(device or "").strip().lower() == "cpu" else FIX_CRASH
PROBE_MARK = "@@A2179_PROBE@@ "

FIX_INSTALL = ("сначала torch под свою ОС и CUDA (команда с pytorch.org), потом "
               "pip install -r requirements-embed.txt, подробности в README, раздел «Эмбеддер не работает»")


class EmbedderUnavailable(RuntimeError):
    def __init__(self, stage: str, cause: str, fix: str):
        self.stage, self.cause, self.fix = stage, cause, fix
        super().__init__(f"{stage}: {cause}")

    def traceback_text(self) -> str:
        if getattr(self, "child_traceback", None):
            return self.child_traceback.strip()
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
                                   "Redistributable x64 и переустанови torch. Проверь, что Python 64 bit")
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

    def __init__(self, model_name: str = DEFAULT_MODEL, local_only: bool = False, pos=None, neg=None,
                 device: str = "auto", backend: str = "torch"):
        self.cpu_guard = cpu_guard(device)  # before torch is imported
        try:
            import_torch_first()
            st = importlib.import_module("sentence_transformers")
            ST = st.SentenceTransformer
        except Exception as e:
            raise classify_import_error(e) from e
        self.model_name = model_name
        self.device = resolve_device(device)
        if self.device == "cpu":
            patch_cuda_capture_checks(self.cpu_guard)  # before the model is built
        self.backend = (backend or "torch").strip().lower()
        t0 = time.monotonic()
        try:
            kw = {"device": self.device}
            if self.backend not in ("", "torch"):
                kw["backend"] = self.backend  # sentence-transformers >= 3.2: "onnx" or "openvino"
            if local_only:
                kw["local_files_only"] = True
            try:
                self.model = ST(model_name, **kw)
            except TypeError as e:
                if "backend" in str(e) and "backend" in kw:
                    raise EmbedderUnavailable("load_failed", _short(e),
                                              "embedder.backend нужен sentence-transformers>=3.2: "
                                              "pip install -U \"sentence-transformers>=3.2,<6\"") from e
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


def import_torch_first():
    """Import torch (if installed) before sentence_transformers: sentence_transformers pulls in
    sklearn, whose OpenMP runtime (libgomp via loky/threadpoolctl) clashes with torch's when it
    is loaded first. A failing torch import propagates (classified by the caller)."""
    if "torch" not in sys.modules and importlib.util.find_spec("torch") is not None:
        importlib.import_module("torch")


def cpu_guard(requested) -> dict:
    """embedder.device=cpu: hide CUDA from torch, so nothing can initialise the CUDA runtime.

    Seen on WSL2 (torch 2.15 cu132, transformers 5.23, Python 3.14): even with device='cpu',
    transformers asks torch.cuda.is_current_stream_capturing() (utils/import_utils.py
    is_cuda_stream_capturing, via is_tracing), which initialises CUDA and segfaults there.
    CUDA_VISIBLE_DEVICES="" works only before torch is imported, so it is set only then;
    otherwise patch_cuda_capture_checks() is the guard. Returns what was done."""
    info = {"cuda_hidden": False, "patched": []}
    if str(requested or "").strip().lower() == "cpu":
        if "torch" not in sys.modules:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            info["cuda_hidden"] = True
        else:
            info["cuda_hidden"] = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    return info


def _never_capturing():
    return False


_never_capturing._a2179_cpu = True


def patch_cuda_capture_checks(info: dict = None) -> list:
    """For a cpu run: make the CUDA graph capture checks answer False without touching CUDA.
    Patches torch.cuda.is_current_stream_capturing (every caller goes through it) and
    transformers.utils.import_utils.is_cuda_stream_capturing (exists in transformers 4.x and
    5.x, is_tracing looks it up there). Each one only if present, never raises."""
    done = []
    torch = sys.modules.get("torch")
    try:
        tc = getattr(torch, "cuda", None)
        if tc is not None and hasattr(tc, "is_current_stream_capturing") \
                and not getattr(tc.is_current_stream_capturing, "_a2179_cpu", False):
            tc.is_current_stream_capturing = _never_capturing
            done.append("torch.cuda.is_current_stream_capturing")
    except Exception:
        pass
    iu = sys.modules.get("transformers.utils.import_utils")
    try:
        if iu is not None and hasattr(iu, "is_cuda_stream_capturing") \
                and not getattr(iu.is_cuda_stream_capturing, "_a2179_cpu", False):
            iu.is_cuda_stream_capturing = _never_capturing
            done.append("transformers.utils.import_utils.is_cuda_stream_capturing")
    except Exception:
        pass
    if info is not None:
        info.setdefault("patched", []).extend(done)
    return done


def resolve_device(requested: str = "auto") -> str:
    """auto -> cuda if torch.cuda.is_available() else cpu. Anything else is passed through."""
    req = str(requested or "auto").strip().lower()
    if req != "auto":
        return req
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import_torch_first()
            torch = sys.modules.get("torch")
        except Exception:
            torch = None
    try:
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _emb_cfg(cfg: dict) -> dict:
    e = dict(cfg.get("embedder") or {})
    e.setdefault("model", DEFAULT_MODEL)
    e.setdefault("device", "auto")
    e.setdefault("backend", "torch")
    return e


def run_probe(cfg: dict, mode: str = "probe", timeout_s: float = None, cmd=None) -> dict:
    """Load + test encode in a child process. Never raises, never crashes the caller.

    Returns {"ok": bool, "device", "dim", "load_s", "stage", "cause", "fix", "traceback",
    "report": [lines printed by the child], "returncode", "stderr_tail"}.
    cmd: argv override (tests use it to simulate crashing or hanging children)."""
    e = _emb_cfg(cfg)
    timeout = float(timeout_s if timeout_s is not None else e.get("probe_timeout_s", 600))
    sub = {"embedder": e, "thresholds": cfg.get("thresholds") or {}}
    root = Path(__file__).resolve().parent.parent
    argv = cmd or [sys.executable, "-X", "faulthandler", "-m", "a2179.embed_probe", mode, json.dumps(sub)]
    env = dict(os.environ, PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
    if str(e.get("device") or "").strip().lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # see cpu_guard: the child never sees a GPU
    env["PYTHONPATH"] = os.pathsep.join([str(root)] + [x for x in [os.environ.get("PYTHONPATH")] if x])
    res = {"ok": False, "report": [], "returncode": None, "stderr_tail": ""}
    t0 = time.monotonic()
    try:
        cp = subprocess.run(argv, cwd=str(root), env=env, capture_output=True, text=True, timeout=timeout,
                            errors="replace")
    except subprocess.TimeoutExpired as te:
        res.update(stage="probe_timeout",
                   cause=f"нет ответа за {timeout:.0f} с (embedder.probe_timeout_s), процесс убит",
                   fix="первый запуск качает модель ~470 МБ: подними embedder.probe_timeout_s. Если зависает "
                       "всегда, это драйвер или torch: " + fix_crash(e.get("device")),
                   stderr_tail=_tail(te.stderr))
        return res
    except Exception as ex:  # could not even start the child
        res.update(stage="probe_failed", cause=_short(ex), fix="проверь python и права на папку проекта")
        return res
    res["probe_s"] = round(time.monotonic() - t0, 2)
    res["returncode"] = cp.returncode
    res["stderr_tail"] = _tail(cp.stderr)
    data = None
    for line in (cp.stdout or "").splitlines():
        if line.startswith(PROBE_MARK):
            try:
                data = json.loads(line[len(PROBE_MARK):])
            except ValueError:
                data = None
        else:
            res["report"].append(line)
    # killed by a signal (POSIX: negative code) or a Windows crash code (0xC0000005 and friends)
    if cp.returncode < 0 or cp.returncode > 255:
        rc = cp.returncode
        try:
            how = f"сигнал {signal.Signals(-rc).name}" if rc < 0 else f"код {rc} ({rc & 0xFFFFFFFF:#x})"
        except ValueError:
            how = f"код {rc}"
        late = " уже после успешной проверки (при выходе)" if data and data.get("ok") else ""
        res.update(stage="crashed", cause=f"процесс проверки умер: {how}{late}", fix=fix_crash(e.get("device")),
                   device=(data or {}).get("device"))
        return res
    if data is None or cp.returncode not in (0, 3):
        last = (res["stderr_tail"].strip().splitlines() or [""])[-1][:240]
        res.update(stage="probe_failed", cause=f"процесс проверки не вернул результат (код {cp.returncode}) {last}".strip(),
                   fix="пришли вывод python monitor.py --check-embedder")
        return res
    res.update(data)
    return res


def _tail(text, n=4000) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return (text or "")[-n:]


def _probe_error(r: dict) -> EmbedderUnavailable:
    err = EmbedderUnavailable(r.get("stage") or "probe_failed", r.get("cause") or "?", r.get("fix") or FIX_CRASH)
    err.child_traceback = (r.get("traceback") or "") + (("\n--- stderr ---\n" + r["stderr_tail"])
                                                        if r.get("stderr_tail") else "")
    return err


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
    from .target import target_of
    tg = target_of(cfg)
    if want == "off":
        return TfidfSim(tg.positive, tg.negative), "tfidf", None
    e = _emb_cfg(cfg)
    try:
        device = e["device"]
        if e.get("probe", True):
            r = run_probe(cfg)
            if log:
                ev(log, "embedder_probe", logging.INFO if r.get("ok") else logging.WARNING,
                   ok=bool(r.get("ok")), stage=r.get("stage"), device=r.get("device"), dim=r.get("dim"),
                   returncode=r.get("returncode"), probe_s=r.get("probe_s"), stderr_tail=r.get("stderr_tail", "")[-2000:])
            if not r.get("ok"):
                raise _probe_error(r)
            device = r.get("device") or device  # load in-process on exactly the device that passed
        sim = EmbedSim(e["model"], bool(e.get("local_only", False)), tg.positive, tg.negative,
                       device=device, backend=e.get("backend"))
        if log:
            ev(log, "embedder_ready", model=sim.model_name, load_s=round(sim.load_s, 2), device=sim.device,
               backend=sim.backend)
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
        return TfidfSim(tg.positive, tg.negative), "tfidf", err


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


def check_embedder(cfg: dict, out=sys.stdout, isolate: bool = True, cmd=None) -> int:
    """Print a diagnostic report. Returns 0 if the embedder works, 3 if not.
    isolate=True: the report is produced by a child process (run_probe "check"), so a native
    crash there is reported instead of killing this process."""
    if not isolate:
        return check_inprocess(cfg, out)[0]
    p = lambda *a: print(*a, file=out)  # noqa: E731
    r = run_probe(cfg, "check", cmd=cmd)
    for line in r.get("report", []):
        p(line)
    if r.get("ok"):
        return 0
    if r.get("stage") in ("crashed", "probe_timeout", "probe_failed"):
        err = _probe_error(r)
        p("== A2179 embedder check ==" if not r.get("report") else "-------------------")
        p(f"probe:      FAILED, {STAGES[err.stage]} ({err.stage}): {err.cause}")
        if r.get("stderr_tail"):
            p("---- child stderr (tail) ----")
            p(r["stderr_tail"].rstrip())
            p("-----------------------------")
        p(err.console())
        p("mode:       tfidf (fallback)")
    return 3


def check_inprocess(cfg: dict, out=sys.stdout):
    """The report itself (runs in the probe child). Returns (rc, info dict)."""
    p = lambda *a: print(*a, file=out)  # noqa: E731
    e = _emb_cfg(cfg)
    model = e.get("model") or DEFAULT_MODEL
    cache = hf_cache_dir()
    p("== A2179 embedder check ==")
    p(f"python:     {sys.version.split()[0]} ({platform.python_implementation()}, {8 * np.dtype('intp').itemsize} bit)")
    p(f"os:         {platform.system()} {platform.release()} {platform.machine()}")
    for pkg in ("torch", "sentence-transformers", "transformers", "huggingface-hub", "numpy", "tokenizers"):
        p(f"{pkg + ':':<23} {_ver(pkg)}")
    p(f"model:      {model}  (local_only={bool(e.get('local_only', False))}, backend={e.get('backend')})")
    p(f"device:     embedder.device={e.get('device')}")
    p(f"config:     embedder.enabled={e.get('enabled', 'auto')}  -> {embedder_wanted(cfg)}")
    writable = os.access(cache if cache.exists() else cache.parent if cache.parent.exists() else Path.home(), os.W_OK)
    p(f"cache dir:  {cache}  (exists={cache.exists()}, writable={writable})")
    for k in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HTTPS_PROXY", "HTTP_PROXY"):
        if os.environ.get(k):
            p(f"env:        {k}={'set' if 'PROXY' in k else os.environ[k]}")
    t0 = time.monotonic()
    try:
        sim = EmbedSim(model, bool(e.get("local_only", False)), device=e.get("device"), backend=e.get("backend"))
    except EmbedderUnavailable as err:
        p(f"load:       FAILED after {time.monotonic() - t0:.1f} s")
        p("---- traceback ----")
        p(err.traceback_text())
        p("-------------------")
        p(err.console())
        p("mode:       tfidf (fallback)")
        return 3, _err_info(err)
    p(f"load:       ok in {sim.load_s:.1f} s on device {sim.device} (first run includes the ~470 MB download)")
    if sim.device == "cpu":
        g = sim.cpu_guard
        p(f"cpu guard:  CUDA_VISIBLE_DEVICES={'empty' if g.get('cuda_hidden') else 'not changed (torch was already imported)'}, "
          f"patched: {', '.join(g.get('patched') or []) or 'nothing to patch'}")
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
        return 3, _err_info(err)
    th = cfg.get("thresholds", {}).get("embed", {})
    p(f"thresholds: embed pos={th.get('pos')} margin={th.get('margin')} "
      f"({'calibrated' if th.get('calibrated') else 'NOT calibrated: run python eval/run_eval.py --write-thresholds'})")
    p(f"mode:       embed (device {sim.device})")
    return 0, {"ok": True, "device": sim.device, "dim": int(dim), "load_s": round(sim.load_s, 2)}


def _err_info(err: EmbedderUnavailable) -> dict:
    return {"ok": False, "stage": err.stage, "cause": err.cause, "fix": err.fix,
            "traceback": err.traceback_text()[-4000:]}
