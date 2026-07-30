"""Per-stage latency accumulation.

Each stage is timed separately so the report can attribute cost — the point of measuring is
knowing which stage to cut when p95 is unacceptable, not just that the pipeline is slow.
"""

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class StageTimer:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append((time.perf_counter() - start) * 1000.0)

    def percentiles(self, stages: list[str] | None = None) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name in stages or self.samples:
            vals = self.samples.get(name)
            if not vals:
                continue
            p50, p95 = np.percentile(vals, [50, 95])
            out[name] = {
                "n": len(vals),
                "p50_ms": float(p50),
                "p95_ms": float(p95),
                "mean_ms": float(np.mean(vals)),
            }
        return out

    def total_per_query(self, stages: list[str]) -> dict[str, float]:
        """Percentiles of the summed end-to-end time, aligned per query.

        Summing the per-stage p95s would overstate the tail: no single query is required to
        hit the 95th percentile of every stage at once.
        """
        present = [s for s in stages if self.samples.get(s)]
        if not present:
            return {}
        n = min(len(self.samples[s]) for s in present)
        totals = [sum(self.samples[s][i] for s in present) for i in range(n)]
        p50, p95 = np.percentile(totals, [50, 95])
        return {
            "n": n,
            "p50_ms": float(p50),
            "p95_ms": float(p95),
            "mean_ms": float(np.mean(totals)),
        }
