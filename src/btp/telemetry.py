"""Optional CUDA timing/profiling for evaluation; baseline defaults do not use it."""

import resource
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from btp.experiment import write_json


class Telemetry:
    """Record synchronized sampling wall time and non-synchronized phase traces."""

    def __init__(self, torch: Any, output: Path, profile: bool = False) -> None:
        self.torch, self.output, self.profile_enabled = torch, output, profile
        self.counts = Counter()
        self.enqueue_seconds = defaultdict(float)
        self.transfers = Counter()
        self.sampling = {}
        self.before_peak = (0, 0)
        self.in_sampling = False
        self.in_guidance = False
        self.started = time.perf_counter()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """CPU enqueue wall time only; profiler supplies CUDA event durations."""
        start = time.perf_counter()
        with self.torch.autograd.profiler.record_function(name):
            try:
                yield
            finally:
                self.enqueue_seconds[name] += time.perf_counter() - start
                self.counts[name] += 1

    @contextmanager
    def sample(self) -> Iterator[None]:
        """Measure a complete synchronized sampler, including CPU transfers."""
        t = self.torch
        t.cuda.synchronize()
        self.before_peak = (t.cuda.max_memory_allocated(), t.cuda.max_memory_reserved())
        self.sampling["allocated_at_start_bytes"] = t.cuda.memory_allocated()
        t.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        self.in_sampling = True
        try:
            with self.phase("sampling"):
                yield
        finally:
            t.cuda.synchronize()
            self.in_sampling = False
            self.sampling.update(
                wall_seconds=time.perf_counter() - start,
                peak_allocated_bytes=t.cuda.max_memory_allocated(),
                peak_reserved_bytes=t.cuda.max_memory_reserved(),
            )

    def transfer(self, name: str, module: Any, direction: str) -> None:
        """Count tensor payload bytes (not measured PCIe traffic)."""
        seen = set()
        size = 0
        for tensor in list(module.parameters()) + list(module.buffers()):
            key = (tensor.data_ptr(), tensor.numel(), tensor.element_size())
            if key not in seen:
                size += tensor.numel() * tensor.element_size()
                seen.add(key)
        self.transfers[name + "_" + direction + "_bytes"] += size
        self.transfers[direction + "_bytes"] += size
        self.transfers[direction + "_module_moves"] += 1

    def save(self, state: str, error: str | None = None) -> None:
        """Persist counters even for unsuccessful inference."""
        t = self.torch
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        doc = {
            "state": state,
            "error": error,
            "sampling": self.sampling,
            "counts": dict(self.counts),
            "transfers": dict(self.transfers),
            "phase_cpu_enqueue_seconds": dict(self.enqueue_seconds),
            "phase_timing_note": "CPU enqueue durations overlap/nest; not additive synchronized GPU timings",
            "runner_elapsed_seconds": time.perf_counter() - self.started,
            "lifetime_peak_allocated_bytes": max(
                self.before_peak[0], t.cuda.max_memory_allocated()
            ),
            "lifetime_peak_reserved_bytes": max(
                self.before_peak[1], t.cuda.max_memory_reserved()
            ),
            "host_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "gpu_total_bytes": t.cuda.get_device_properties(0).total_memory,
            "profiled": self.profile_enabled,
        }
        write_json(self.output / "telemetry.json", doc)
