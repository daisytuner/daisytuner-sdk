import dace
import requests

from typing import Dict

from daisytuner.benchmarking.cpu_benchmark import CPUBenchmark
from daisytuner.benchmarking.gpu_benchmark import GPUBenchmark


class DeviceMapper:
    def __init__(
        self,
        sdfg: dace.SDFG,
        agent_type: str,
        cpu_benchmark: CPUBenchmark,
        gpu_benchmark: GPUBenchmark,
    ) -> None:
        self._sdfg = sdfg
        self._agent_type = agent_type
        self._cpu_benchmark = cpu_benchmark
        self._gpu_benchmark = gpu_benchmark

    def tune(self) -> dace.SDFG:
        res = self._run_model()
        sdfg_opt = dace.SDFG.from_json(res["schedule"])
        return sdfg_opt

    def _run_model(self) -> Dict:
        from daisytuner.cli import CLI

        user = CLI.user()

        headers = {"Authorization": "Bearer {}".format(user["idToken"])}
        req = requests.post(
            "https://device-mapping-bhqsvyw3sa-uc.a.run.app",
            headers=headers,
            json={
                "sdfg": self._sdfg.to_json(),
                "agent_type": self._agent_type,
                "cpu_benchmark": self._cpu_benchmark.data,
                "gpu_benchmark": self._gpu_benchmark.data,
            },
        )
        if not req.ok:
            raise ValueError("API: ", req.content)

        return req.json()
