import dace
import requests

from typing import Dict

from daisytuner.benchmarking.benchmark import Benchmark


class MapNestModel:
    def __init__(self, device: str, benchmark: Benchmark) -> None:
        self._device = device
        self._benchmark = benchmark

    def predict(self, sdfg: dace.SDFG) -> Dict:
        from daisytuner.cli import CLI

        user = CLI.user()

        headers = {"Authorization": "Bearer {}".format(user["idToken"])}
        req = requests.post(
            "https://embedding-bhqsvyw3sa-uc.a.run.app",
            headers=headers,
            json={
                "map_nest": sdfg.to_json(),
                "device_type": self._device,
                "benchmark": self._benchmark.data,
            },
        )
        if not req.ok:
            raise ValueError("API: ", req.content)

        return req.json()
