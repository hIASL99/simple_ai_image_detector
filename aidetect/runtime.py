"""CPU execution settings.

Two defaults that matter on this class of machine, both measured rather than
assumed:

* Threads. Torch defaults to one thread per *logical* core. On an SMT part the
  two siblings share one vector unit, so the extra threads only add contention:
  measured throughput on a 12-core/24-thread Zen 5 part was 9.7 img/s at 12
  threads and 4.8 img/s at 24.
* bfloat16. Where the CPU has AVX-512-BF16 (or AMX), autocast roughly doubles
  throughput. It is opt-in because it perturbs individual scores.
"""

from __future__ import annotations

import contextlib
import os

import torch


def physical_cores() -> int:
    """Best-effort physical core count, falling back to half the logical count."""
    try:
        with open("/proc/cpuinfo") as fh:
            ids = {line.split(":")[1].strip()
                   for line in fh if line.startswith(("core id", "physical id"))}
        if ids:
            import re
            with open("/proc/cpuinfo") as fh:
                text = fh.read()
            pairs = set(zip(re.findall(r"physical id\s*:\s*(\d+)", text),
                            re.findall(r"core id\s*:\s*(\d+)", text)))
            if pairs:
                return len(pairs)
    except OSError:
        pass
    logical = os.cpu_count() or 4
    return max(1, logical // 2)


def configure_threads(num_threads: int | None = None) -> int:
    """Set torch's intra-op thread count; defaults to the physical core count."""
    n = int(num_threads) if num_threads else physical_cores()
    n = max(1, n)
    torch.set_num_threads(n)
    return n


def cpu_supports_bf16() -> bool:
    try:
        with open("/proc/cpuinfo") as fh:
            flags = fh.read()
    except OSError:
        return False
    return "avx512_bf16" in flags or "amx_bf16" in flags


@contextlib.contextmanager
def autocast_cpu(enabled: bool):
    if not enabled:
        yield
        return
    with torch.autocast("cpu", dtype=torch.bfloat16):
        yield
