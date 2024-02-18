import dace
import copy

from typing import Dict, Any, List, Union
from tqdm import tqdm

from dace.transformation import pass_pipeline as ppl

from daisytuner.benchmarking import CPUBenchmark, GPUBenchmark
from daisytuner.embeddings.map_nest import MapNest
from daisytuner.profiling.helpers import random_arguments
from daisytuner.transfer_tuning.transfer_tuner import TransferTuner


class TransferTuningPipeline(ppl.Pass):

    CATEGORY: str = "Optimization"

    recursive = True

    def __init__(
        self,
        benchmark: Union[CPUBenchmark, GPUBenchmark],
        device: str,
        collection: str = "default",
        topk: int = 3,
    ) -> None:
        super().__init__()
        self._benchmark = benchmark
        self._device = device
        self._collection = collection
        self._topk = topk

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Scopes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        map_nests: List[MapNest] = []
        for state in sdfg.states():
            for node in state.nodes():
                if not isinstance(node, dace.nodes.MapEntry):
                    continue
                if state.entry_node(node) is not None:
                    continue

                map_nest = MapNest(state, node)
                map_nests.append(map_nest)

        print("Tuning map nests...")
        pipeline_results[TransferTuningPipeline.__class__.__name__] = {}
        for map_nest in tqdm(map_nests):
            cutout = map_nest.as_cutout()
            args = random_arguments(cutout)

            tuner = TransferTuner(
                map_nest=map_nest,
                arguments=args,
                device=self._device,
                benchmark=self._benchmark,
                collection=self._collection,
                topk=self._topk,
            )
            opt_sdfg, _ = tuner.tune()

            TransferTuningPipeline.replace_by_cutout(sdfg, map_nest, opt_sdfg)
            pipeline_results[TransferTuningPipeline.__class__.__name__][
                map_nest
            ] = opt_sdfg

    @staticmethod
    def replace_by_cutout(
        sdfg: dace.SDFG, map_nest: MapNest, cutout: dace.SDFG
    ) -> None:
        state = map_nest.state
        nsdfg = state.add_nested_sdfg(
            cutout,
            state,
            map_nest.inputs(wcr_as_input=False),
            map_nest.outputs(),
            name=map_nest.root.map.label + "_opt",
        )

        for iedge in state.in_edges(map_nest.root):
            if iedge.data.data not in nsdfg.in_connectors:
                nsdfg.add_in_connector(iedge.data.data)
            state.add_edge(
                iedge.src,
                iedge.src_conn,
                nsdfg,
                iedge.data.data,
                copy.deepcopy(iedge.data),
            )

        for oedge in state.out_edges(state.exit_node(map_nest.root)):
            if oedge.data.data not in nsdfg.out_connectors:
                nsdfg.add_out_connector(oedge.data.data)
            state.add_edge(
                nsdfg,
                oedge.data.data,
                oedge.dst,
                oedge.dst_conn,
                copy.deepcopy(oedge.data),
            )

        subgraph = state.scope_subgraph(
            map_nest.root, include_entry=True, include_exit=True
        )
        for node in subgraph.nodes():
            state.remove_node(node)

        for edge in subgraph.edges():
            state.remove_edge(edge)
