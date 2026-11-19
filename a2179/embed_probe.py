"""Embedder probe child: python -m a2179.embed_probe probe|check '<json config>'.

Started by embedder.run_probe in a separate process. Loads the model (torch first, then
sentence_transformers) on the configured device, encodes test texts, and prints one line
`@@A2179_PROBE@@ {json}` with ok, device, dim, load_s or the failure stage. If torch or the
driver crashes natively, only this process dies; the monitor sees the signal and falls back.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a2179.embedder import (DEFAULT_MODEL, PROBE_MARK, EmbedderUnavailable, EmbedSim,  # noqa: E402
                            _emb_cfg, _err_info, check_inprocess)


def probe(cfg: dict) -> dict:
    e = _emb_cfg(cfg)
    try:
        t0 = time.monotonic()
        sim = EmbedSim(e.get("model") or DEFAULT_MODEL, bool(e.get("local_only", False)),
                       device=e.get("device"), backend=e.get("backend"))
        a = sim.score("LCD ekran za MacBook Air A2179 2020")
        sim.score("Futrola za MacBook Air A2179")
        return {"ok": True, "device": sim.device, "dim": int(sim.P.shape[1]), "load_s": round(sim.load_s, 2),
                "cuda_hidden": sim.cpu_guard.get("cuda_hidden"), "patched": sim.cpu_guard.get("patched"),
                "total_s": round(time.monotonic() - t0, 2), "sanity_pos": round(float(a[0]), 3)}
    except EmbedderUnavailable as err:
        return _err_info(err)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else "probe"
    cfg = json.loads(argv[1]) if len(argv) > 1 else {}
    if mode == "check":
        rc, info = check_inprocess(cfg, sys.stdout)
    else:
        info = probe(cfg)
        rc = 0
    sys.stdout.flush()
    print(PROBE_MARK + json.dumps(info, ensure_ascii=False), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
