import dace
import platform
import numpy as np

from daisytuner.profiling.metrics.metric import Metric


class Instructions(Metric):
    def __init__(
        self, sdfg: dace.SDFG, hostname: str = platform.node(), cache=None
    ) -> None:
        super().__init__(
            sdfg,
            [
                "INST_RETIRED",
            ],
            "cpu",
            hostname,
            "arm64fx",
            cache=cache,
        )

    def compute(self) -> float:
        counters = self.values()

        volume_branches = 0.0
        for state in self._sdfg.states():
            volume_branches += sum(
                [
                    measurements[0]
                    for thread_id, measurements in counters["INST_RETIRED"][
                        state
                    ].items()
                ]
            )

        metric = volume_branches
        return metric

    def compute_per_thread(self) -> np.ndarray:
        counters = self.values()

        volume_branches = []
        for state in self._sdfg.states():
            volume_branches.append(
                np.array(
                    [
                        measurements[0]
                        for thread_id, measurements in counters["INST_RETIRED"][
                            state
                        ].items()
                    ]
                )
            )

        metric = np.vstack(volume_branches).sum(axis=0, keepdims=False)
        return metric
