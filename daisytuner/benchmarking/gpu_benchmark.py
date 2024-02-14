from __future__ import annotations

import json
import platform

from typing import Dict
from pathlib import Path

from daisytuner.benchmarking.benchmark import Benchmark


class GPUBenchmark(Benchmark):
    def __init__(self, data: Dict) -> None:
        super().__init__(data)
        assert set(data.keys()) == set(
            (
                "devices",
                "arch",
                "compute_capability",
                "l2_cache",
                "memory",
                "SIMD_width",
                "clock_rate",
                "mem_clock_rate",
            )
        )

    @staticmethod
    def measure() -> GPUBenchmark:
        raise NotImplementedError

    @staticmethod
    def from_cache(hostname: str = platform.node()) -> GPUBenchmark:
        cache_path = Path.home() / ".daisytuner" / f"{hostname}.json"
        if cache_path.is_file():
            with open(cache_path, "r") as handle:
                data = json.load(handle)["gpu"]
                return GPUBenchmark(data)

        return None
