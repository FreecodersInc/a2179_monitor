"""Print what this runner has, as one JSON line. Run in its own process: importing a broken
CUDA torch can kill the interpreter, and then the caller just records the crash."""
import json
import os
import platform
import sys


def main():
    info = {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "torch": None, "cuda_available": False, "cuda_version": None,
            "device_name": None, "device_count": 0, "sentence_transformers": None, "transformers": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:  # ImportError or a broken build
        info["torch_error"] = f"{type(e).__name__}: {e}"[:300]
    for mod in ("sentence_transformers", "transformers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:
            info[f"{mod}_error"] = f"{type(e).__name__}: {e}"[:300]
    print(json.dumps(info))


if __name__ == "__main__":
    sys.exit(main())
