import dace

from typing import Dict, Any

from dace.transformation import pass_pipeline as ppl
from dace.transformation.dataflow import MapCollapse


class MapCompactForm(ppl.Pass):
    """
    Squashes maps to a compact form, for which many transformations are designed.
    """

    CATEGORY: str = "Normalization"

    recursive = True

    def __init__(self) -> None:
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Scopes

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        applied = sdfg.apply_transformations_repeated(MapCollapse, validate=False)
        if applied > 0:
            return applied
        else:
            return None
