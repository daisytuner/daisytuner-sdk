from __future__ import annotations

from abc import ABC

from typing import Dict


class Benchmark(ABC):
    def __init__(self, data: Dict) -> None:
        self.data = data

    @staticmethod
    def measure() -> Benchmark:
        raise NotImplementedError

    @staticmethod
    def from_cache() -> Benchmark:
        raise NotImplementedError
