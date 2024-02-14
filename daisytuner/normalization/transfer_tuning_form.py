import dace

from typing import Dict, Any

from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import MapExpansion


class TransferTuningForm(ppl.Pass):

    CATEGORY: str = "Normalization"

    recursive = True

    def __init__(self) -> None:
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Scopes

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        applied = sdfg.apply_transformations_repeated(MapExpansion)
        for state in sdfg.states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.MapEntry):
                    continue

                node.schedule = dace.ScheduleType.Sequential
                node.collapse = 1

        sdfg.openmp_sections = False
        return None
