from __future__ import annotations

import re
import json
import platform
import subprocess

from typing import Dict
from tqdm import tqdm
from pathlib import Path

from daisytuner.benchmarking.benchmark import Benchmark


class CPUBenchmark(Benchmark):
    def __init__(self, data: Dict) -> None:
        super().__init__(data)
        assert set(data.keys()) == set(
            (
                "arch",
                "num_sockets",
                "cores_per_socket",
                "threads_per_core",
                "l2_cache",
                "l3_cache",
                "stream_copy",
                "stream_load",
                "stream_store",
                "stream_triad",
                "peakflops",
                "peakflops_avx",
            )
        )

    @staticmethod
    def measure() -> CPUBenchmark:
        from daisytuner.profiling.likwid_helpers import cpu_codename, cpu_topology

        arch = cpu_codename()
        topo = cpu_topology()

        data = {}
        data["arch"] = arch
        data["num_sockets"] = topo["numSockets"]
        data["cores_per_socket"] = topo["numCoresPerSocket"]
        data["threads_per_core"] = topo["numThreadsPerCore"]
        data["l2_cache"] = int(topo["cacheLevels"][2]["size"] / 1000)
        data["l3_cache"] = int(topo["cacheLevels"][3]["size"] / 1000)

        num_cpus = (
            data["threads_per_core"] * data["cores_per_socket"] * data["num_sockets"]
        )

        print("Executing Benchmarks...")
        data.update(CPUBenchmark._stream_benchmark(num_cpus))
        data.update(CPUBenchmark._peakflops_benchmark(num_cpus))

        # Cache benchmark
        hostname = platform.node()
        cache_data = {"cpu": data, "gpu": None}
        cache_path = Path.home() / ".daisytuner" / f"{hostname}.json"
        if cache_path.is_file():
            with open(cache_path, "r") as handle:
                cache_data["gpu"] = json.load(handle)["gpu"]

        with open(cache_path, "w") as handle:
            json.dump(cache_data, handle)

        return CPUBenchmark(data)

    @staticmethod
    def from_cache(hostname: str = platform.node()) -> CPUBenchmark:
        cache_path = Path.home() / ".daisytuner" / f"{hostname}.json"
        if cache_path.is_file():
            with open(cache_path, "r") as handle:
                data = json.load(handle)["cpu"]
                return CPUBenchmark(data)

        return None

    @staticmethod
    def _stream_benchmark(num_cores: int) -> Dict:
        stream = {}
        for test in tqdm(["load", "store", "copy", "triad"]):
            process = subprocess.Popen(
                ["likwid-bench", f"-t{test}", f"-WN:2GB:{num_cores}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            stdout, stderr = process.communicate()
            res = re.findall(r"MByte/s:\t\t\d+\.\d+", stdout)
            if not res:
                raise ValueError(stderr)
            stream[f"stream_{test}"] = float(re.findall(r"\d+\.\d+", res[0])[0])

        return stream

    @staticmethod
    def _peakflops_benchmark(num_cores: int) -> Dict:
        peakflops = {}
        for name, test in tqdm(
            [("peakflops", "peakflops"), ("peakflops_avx", "peakflops_avx_fma")]
        ):
            process = subprocess.Popen(
                ["likwid-bench", f"-t{test}", f"-WN:360kB:{num_cores}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            stdout, stderr = process.communicate()
            res = re.findall(r"MFlops/s:\t\t\d+\.\d+", stdout)
            if not res:
                raise ValueError(stderr)
            peakflops[name] = float(re.findall(r"\d+\.\d+", res[0])[0])

        return peakflops
