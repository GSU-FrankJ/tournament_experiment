"""Shared paths, strict JSON/JSONL writers and resource probes for the T3 pilot.

The numerical core (GameSpec, CurriculumPPO, collect_batch math, DP-BR verifier)
is imported from this repository root (W is retained as a legacy variable name). Run every
entry point with ``python -B`` so importing W does not write bytecode into it.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
W = ROOT
CHECKOUT = E.parent.parent          # git checkout that contains this experiment directory
PYTHON = Path(sys.executable)
PLAN = ROOT / "MultiStage" / "three_stage" / "T3_IMPLEMENTATION_PLAN_20260924.md"
W_SOURCES = ("envs/curriculum_env.py", "agents/ppo_curriculum.py", "run/run_final_dp_br.py",
             "utils/dp_br_verifier.py", "utils/theory_multistage.py")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

if str(W) not in sys.path:
    sys.path.insert(0, str(W))
if str(E) not in sys.path:
    sys.path.insert(1, str(E))


# ---------------------------------------------------------------------------
# Strict JSON
# ---------------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    """Convert numpy/scalars recursively; non-finite floats become None.

    Args:
        obj: Arbitrary nested object.

    Returns:
        A structure accepted by ``json.dumps(allow_nan=False)``.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dumps(obj: Any, indent: Optional[int] = None) -> str:
    """Strict JSON text (NaN/Infinity are converted to null first)."""
    return json.dumps(to_jsonable(obj), indent=indent, allow_nan=False, ensure_ascii=False)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON through a temporary file and ``os.replace`` (no half-written file)."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(dumps(obj, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: Any) -> None:
    """Append one strict-JSON line and flush it to disk immediately."""
    with open(path, "a") as f:
        f.write(dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read JSONL, tolerating one truncated final line.

    Args:
        path: File path.

    Returns:
        ``(records, info)`` where info records ``exists``, ``n_lines`` and
        ``truncated_last_line`` (a non-parseable last line is dropped, never an
        earlier complete line).

    Raises:
        ValueError: If a line other than the last one does not parse.
    """
    info: Dict[str, Any] = {"exists": Path(path).exists(), "n_lines": 0, "truncated_last_line": False}
    if not info["exists"]:
        return [], info
    text = Path(path).read_text()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                info["truncated_last_line"] = True
                break
            raise ValueError(f"{path}: corrupt JSONL line {i + 1}")
    info["n_lines"] = len(out)
    return out, info


def now_iso() -> str:
    """Local timestamp with timezone offset."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _git(args: List[str], cwd: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(cwd)] + args,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def provenance() -> Dict[str, Any]:
    """Git versions, git status of the imported W sources, and source file stats.

    The W modules used here are untracked in W, so ``git diff`` does not cover
    them; their size and mtime are recorded as plain provenance.
    """
    import platform

    import numpy as np
    import torch

    def stat(p: Path) -> Dict[str, Any]:
        try:
            st = p.stat()
            return {"path": str(p), "size": st.st_size,
                    "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))}
        except OSError as exc:
            return {"path": str(p), "error": str(exc)}

    return {
        "checkout": str(CHECKOUT),
        "checkout_git_head": _git(["rev-parse", "HEAD"], CHECKOUT),
        "checkout_git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], CHECKOUT),
        "checkout_git_status_experiment_dir": _git(["status", "--porcelain", "--", str(E)], CHECKOUT),
        "root_git_head": _git(["rev-parse", "HEAD"], ROOT),
        "w_git_head": _git(["rev-parse", "HEAD"], W),
        "w_git_status_sources": _git(["status", "--porcelain", "--"] + list(W_SOURCES), W),
        "w_git_diff_stat_sources": _git(["diff", "--stat", "--"] + list(W_SOURCES), W),
        "w_sources": [stat(W / s) for s in W_SOURCES],
        "experiment_sources": [stat(p) for p in sorted(E.glob("*.py"))]
        + [stat(p) for p in sorted((E / "tests").glob("*.py"))],
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "hostname": os.uname().nodename,
        "paths": {"ROOT": str(ROOT), "W": str(W), "E": str(E), "plan": str(PLAN)},
    }


def thread_settings() -> Dict[str, Any]:
    """Thread environment and torch thread count."""
    import torch

    return {"env": {k: os.environ.get(k) for k in THREAD_ENV},
            "torch_num_threads": torch.get_num_threads()}


# ---------------------------------------------------------------------------
# Resources (Linux /proc/self/status, bytes)
# ---------------------------------------------------------------------------

_KB = re.compile(r"^(VmRSS|VmHWM):\s+(\d+)\s+kB", re.M)


def proc_mem() -> Dict[str, Optional[int]]:
    """Current VmRSS and VmHWM in bytes (None + reason if unavailable)."""
    try:
        text = Path("/proc/self/status").read_text()
    except OSError as exc:
        return {"rss_bytes": None, "hwm_bytes": None, "reason": f"/proc unavailable: {exc}"}
    vals = {m.group(1): int(m.group(2)) * 1024 for m in _KB.finditer(text)}
    return {"rss_bytes": vals.get("VmRSS"), "hwm_bytes": vals.get("VmHWM")}


class ResourceTracker:
    """Wall/CPU/RSS per named segment; peaks via VmHWM reset (clear_refs 5).

    At every segment start the high-water mark since the previous reset is
    folded into all open segments, then reset, so nested segments (a dev
    verifier call inside training) keep correct peaks. If the reset is not
    permitted, the process-lifetime VmHWM is reported with ``peak_method`` set
    accordingly (an upper bound, never 0).
    """

    def __init__(self) -> None:
        self.open: List[Dict[str, Any]] = []
        self.process_peak: int = proc_mem().get("hwm_bytes") or 0
        self.can_reset = self._reset()
        self.peak_method = "VmHWM_reset_per_segment" if self.can_reset else "lifetime_VmHWM"

    @staticmethod
    def _reset() -> bool:
        try:
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5")
            return True
        except OSError:
            return False

    def _fold(self) -> None:
        hwm = proc_mem().get("hwm_bytes")
        if hwm is None:
            return
        for seg in self.open:
            seg["_peak"] = max(seg["_peak"], hwm)
        self.process_peak = max(self.process_peak, hwm)

    def current_process_peak(self) -> int:
        """Process peak RSS so far (bytes), including the current open interval."""
        self._fold()
        return self.process_peak

    @contextmanager
    def segment(self, name: str, **extra: Any) -> Iterator[Dict[str, Any]]:
        """Measure a code block; the yielded dict is filled on exit."""
        self._fold()
        if self.can_reset:
            self._reset()
        before = proc_mem()
        seg: Dict[str, Any] = {"segment": name, **extra, "rss_before_bytes": before.get("rss_bytes"),
                               "_peak": before.get("rss_bytes") or 0,
                               "_t0": time.perf_counter(), "_c0": time.process_time()}
        self.open.append(seg)
        try:
            yield seg
        finally:
            self._fold()
            self.open.remove(seg)
            after = proc_mem()
            seg["wall_sec"] = time.perf_counter() - seg.pop("_t0")
            seg["cpu_sec"] = time.process_time() - seg.pop("_c0")
            seg["rss_after_bytes"] = after.get("rss_bytes")
            peak = seg.pop("_peak")
            if not self.can_reset:
                peak = after.get("hwm_bytes")
            seg["rss_peak_bytes"] = peak if peak else None
            if not peak:
                seg["rss_peak_reason"] = "VmHWM unavailable"
            seg["peak_method"] = self.peak_method
