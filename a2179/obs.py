"""Observability: JSON-lines logging and a per-run metrics file."""
import json
import logging
import logging.handlers
import statistics
import time
from collections import defaultdict
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            d.update(extra)
        return json.dumps(d, ensure_ascii=False)


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("a2179")
    for h in list(log.handlers):
        h.close()
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "monitor.jsonl", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(JsonFormatter())
    sh.setLevel(logging.DEBUG if verbose else logging.ERROR)  # full detail is in the log file
    log.addHandler(fh)
    log.addHandler(sh)
    log.propagate = False
    return log


def ev(log: logging.Logger, event: str, level: int = logging.INFO, **fields):
    log.log(level, event, extra={"fields": fields})


class Metrics:
    def __init__(self):
        self.started = time.time()
        self.counters = defaultdict(int)
        self.latency_ms = defaultdict(list)
        self.status = defaultdict(lambda: defaultdict(int))
        self.notes = []

    def inc(self, key: str, n: int = 1):
        self.counters[key] += n

    def fetch(self, host: str, status, ms: float, nbytes: int = 0):
        self.counters[f"fetch.{host}.requests"] += 1
        self.counters["fetch.requests"] += 1
        self.counters["fetch.bytes"] += nbytes
        self.status[host][str(status)] += 1
        self.latency_ms[host].append(ms)
        ok = isinstance(status, int) and 200 <= status < 300
        if not ok:
            self.counters[f"fetch.{host}.errors"] += 1
            self.counters["fetch.errors"] += 1

    def snapshot(self) -> dict:
        lat = {}
        for host, xs in self.latency_ms.items():
            xs = sorted(xs)
            lat[host] = {
                "n": len(xs),
                "avg_ms": round(statistics.fmean(xs), 1),
                "p95_ms": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 1),
                "max_ms": round(xs[-1], 1),
            }
        return {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started)),
            "duration_s": round(time.time() - self.started, 2),
            "counters": dict(sorted(self.counters.items())),
            "http_status": {h: dict(v) for h, v in self.status.items()},
            "latency": lat,
            "notes": self.notes,
        }

    def write(self, data_dir: Path, dry_run: bool = False) -> dict:
        snap = self.snapshot()
        data_dir.mkdir(parents=True, exist_ok=True)
        name = "metrics_dryrun.json" if dry_run else "metrics.json"
        (data_dir / name).write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        if dry_run:
            return snap
        with open(data_dir / "metrics_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(snap, ensure_ascii=False) + "\n")
        return snap
