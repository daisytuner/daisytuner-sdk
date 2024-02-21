import copy
import dace
import math
import json
import copy
import requests
import numpy as np

from tqdm import tqdm
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment

from dace.transformation import PatternTransformation

from daisytuner.benchmarking.benchmark import Benchmark
from daisytuner.embeddings.map_nest import MapNest
from daisytuner.embeddings.map_nest_model import MapNestModel
from daisytuner.profiling.helpers import measure_safe


class TransferTuner:
    def __init__(
        self,
        map_nest: MapNest,
        arguments: Dict,
        device: str,
        benchmark: Benchmark,
        collection: str = "default",
        topk: int = 3,
    ) -> None:
        self._map_nest = map_nest
        self._arguments = arguments
        self._device = device
        self._benchmark = benchmark
        self._model = MapNestModel(self._device, self._benchmark)
        self._topk = topk
        self._collection = collection

        self._nearest_neighbors = self._query_nearest_neighbors()

    def tune(self) -> Tuple[dace.SDFG, Dict]:
        print("Transfer tuning...")

        # Default schedule
        cutout = self._map_nest.as_cutout()
        if self._device == "gpu":
            cutout = TransferTuner.default_gpu_schedule(cutout)

        # Measure base runtime
        if self._device == "gpu":
            cutout_measure = TransferTuner.as_gpu_schedule(cutout)
        else:
            cutout_measure = cutout

        cutout_measure.instrument = dace.InstrumentationType.Timer

        result = measure_safe(cutout_measure, arguments=copy.deepcopy(self._arguments))
        if result is None:
            raise ValueError(
                "Transfer Tuner: Failed to measure initial runtime of map nest"
            )

        report, initial_process_time = result
        initial_runtime = next(
            next(report.durations[(0, -1, -1)].values().__iter__()).values().__iter__()
        )[0]

        cutout_measure.instrument = dace.InstrumentationType.No_Instrumentation
        print("Initial runtime: ", initial_runtime)

        # Test nearest neighbors
        best_candidate = cutout
        best_runtime = initial_runtime
        best_process_time = initial_process_time
        best_nn = None
        for nn in tqdm(self._nearest_neighbors):
            candidate = self._map_nest.as_cutout()
            # Align schedules
            for state in candidate.states():
                for node in state.nodes():
                    if not isinstance(node, dace.nodes.MapEntry):
                        continue

                    node.schedule = dace.ScheduleType.Sequential
                    node.collapse = 1

            source = dace.SDFG.from_json(
                json.loads(json.loads(nn["schedule"]["normal_form"]))
            )
            success = self._transfer_tune(
                source=source,
                target=candidate,
                transformations=nn["schedule"]["transformations"],
            )
            if not success:
                continue

            if self._device == "gpu":
                candidate_measure = TransferTuner.as_gpu_schedule(candidate)
            else:
                candidate_measure = candidate

            candidate_measure.instrument = dace.InstrumentationType.Timer

            report = measure_safe(
                candidate_measure,
                arguments=copy.deepcopy(self._arguments),
                timeout=best_process_time,
            )
            if report is not None:
                report, process_time = report
                runtime = next(
                    next(report.durations[(0, -1, -1)].values().__iter__())
                    .values()
                    .__iter__()
                )[0]
            else:
                runtime, process_time = (math.inf, math.inf)

            candidate_measure.instrument = dace.InstrumentationType.No_Instrumentation

            print("Neighbor: ", runtime)
            if best_runtime / (runtime + 1e-9) > 1.1:
                best_runtime = runtime
                best_process_time = process_time
                best_candidate = candidate
                best_nn = nn

        return best_candidate, best_nn

    def _query_nearest_neighbors(self) -> List[Dict]:
        from daisytuner.cli import CLI

        user = CLI.user()

        headers = {"Authorization": "Bearer {}".format(user["idToken"])}
        req = requests.post(
            "https://nearest-neighbors-bhqsvyw3sa-uc.a.run.app",
            headers=headers,
            json={
                "map_nest": self._map_nest.as_cutout().to_json(),
                "device_type": self._device,
                "collection_id": self._collection,
                "topk": self._topk,
                "benchmark": self._benchmark.data,
            },
        )
        if not req.ok:
            raise ValueError("API: ", req.content)

        return req.json()

    def _transfer_tune(
        self, source: dace.SDFG, target: dace.SDFG, transformations: List[Dict]
    ) -> bool:
        applied_transformations = []
        for trans in transformations:
            source_node_embeddings = self._model.predict(source)["node_embeddings"]
            target_node_embeddings = self._model.predict(target)["node_embeddings"]

            # Compute subgraph matching
            subgraph = trans["_subgraph"]
            source_node_ids = list(subgraph.values())
            source_nodes = []
            target_nodes = []
            cost_matrix = np.zeros((len(source_node_ids), len(target_node_embeddings)))
            for i in range(cost_matrix.shape[0]):
                source_node = source.start_state.node(source_node_ids[i])
                source_node_emb = np.array(
                    source_node_embeddings[str(source_node_ids[i])]
                )
                source_nodes.append(source_node)

                for j, target_node in enumerate(target.start_state.nodes()):
                    target_node_emb = np.array(
                        target_node_embeddings[
                            str(target.start_state.node_id(target_node))
                        ]
                    )
                    if i == 0:
                        target_nodes.append(target_node)

                    if type(source_node) != type(target_node):
                        cost_matrix[i, j] = np.inf
                    else:
                        cost_matrix[i, j] = np.linalg.norm(
                            source_node_emb - target_node_emb, ord=2.0
                        )

            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            except:
                continue

            pattern_nodes = list(subgraph.keys())
            target_subgraph = {}
            for i in range(len(row_ind)):
                pattern_node = pattern_nodes[row_ind[i]]
                source_node = source_nodes[row_ind[i]]
                target_node = target_nodes[col_ind[i]]
                target_subgraph[pattern_node] = target.start_state.node_id(target_node)

            # Instantiate transformation
            trans_target = copy.deepcopy(trans)
            trans_target["_subgraph"] = target_subgraph

            # Replace sdfg-specific options
            if self._align_transformation(target, trans_target):
                xform_target = PatternTransformation.from_json(trans_target)
                xform_target._sdfg = target
                xform_target.state_id = target.node_id(target.start_state)

                # Apply transformation
                if xform_target.can_be_applied(
                    target.start_state, sdfg=target, expr_index=xform_target.expr_index
                ):
                    xform_target.apply_pattern(append=True)
                    applied_transformations.append(xform_target.to_json())

            xform_source = PatternTransformation.from_json(trans)
            xform_source._sdfg = source
            xform_source.state_id = source.node_id(source.start_state)
            xform_source.apply(source.start_state, source)

        return len(applied_transformations) > 0

    def _align_transformation(self, sdfg: dace.SDFG, trans: Dict) -> bool:
        subgraph = trans["_subgraph"]
        transformation = trans["transformation"]

        if transformation == "StripMining":
            map_entry = sdfg.start_state.node(subgraph["0"])
            dim_idx = trans["dim_idx"]
            tile_size = int(trans["tile_size"])

            start, stop, step = map_entry.map.range[dim_idx]
            map_extend = dace.symbolic.int_floor((stop + 1 - start), step)
            try:
                map_extend = dace.symbolic.evaluate(map_extend, symbols=sdfg.constants)
                if tile_size > map_extend:
                    return False

                divides_evenly = map_extend / tile_size
                trans["divides_evenly"] = divides_evenly.is_integer
            except:
                trans["divides_evenly"] = False
        elif transformation == "MapTiling":
            map_entry = sdfg.start_state.node(subgraph["0"])
            tile_size = int(trans["tile_sizes"][0])

            trans["tile_trivial"] = False

            start, stop, step = map_entry.map.range[0]
            map_extend = dace.symbolic.int_floor((stop + 1 - start), step)
            try:
                map_extend = dace.symbolic.evaluate(map_extend, symbols=sdfg.constants)
                if tile_size > map_extend:
                    return False

                divides_evenly = map_extend / tile_size
                trans["divides_evenly"] = divides_evenly.is_integer
            except:
                trans["divides_evenly"] = False
        elif transformation == "Vectorization":
            map_entry = sdfg.start_state.node(subgraph["0"])
            start, stop, step = map_entry.map.range[-1]
            map_extend = dace.symbolic.int_floor((stop + 1 - start), step)

            try:
                map_extend = dace.symbolic.evaluate(map_extend, symbols=sdfg.constants)
                divisor = map_extend / int(trans["vector_len"])
                divides_evenly = divisor.is_integer
                trans["postamble"] = not divides_evenly
            except:
                trans["postamble"] = True

            trans["preamble"] = False

        return True

    @staticmethod
    def default_gpu_schedule(cutout: dace.SDFG) -> dace.SDFG:
        schedule = copy.deepcopy(cutout)
        for node in schedule.start_state.nodes():
            if (
                isinstance(node, dace.nodes.MapEntry)
                and schedule.start_state.entry_node(node) is None
            ):
                node.map.schedule = dace.ScheduleType.GPU_Device

        return schedule

    @classmethod
    def as_gpu_schedule(cls, cutout: dace.SDFG):
        cutout_ = copy.deepcopy(cutout)
        true_state = cutout_.start_state

        arrays = list(cutout_.arrays)
        for dnode in true_state.data_nodes():
            dnode.data = "device_" + dnode.data
        for edge in true_state.edges():
            if edge.data.data is not None:
                edge.data.data = "device_" + edge.data.data

        init_state = cutout_.add_state_before(true_state)
        for item in arrays:
            host_array = item
            host_desc = cutout_.arrays[host_array]
            host_node = init_state.add_access(host_array)

            device_array = "device_" + item
            if device_array not in cutout_.arrays:
                device_desc = copy.deepcopy(host_desc)
                device_desc.storage = dace.StorageType.GPU_Global
                device_desc.transient = True
                cutout_.add_datadesc(device_array, device_desc)
            else:
                device_desc = cutout_.arrays[device_array]

            device_node = init_state.add_access(device_array)

            init_state.add_edge(
                host_node,
                None,
                device_node,
                None,
                dace.Memlet.from_array(host_array, host_desc),
            )

        exit_state = cutout_.add_state_after(true_state)
        for item in arrays:
            host_array = item
            host_desc = cutout_.arrays[host_array]
            host_node = exit_state.add_access(host_array)

            device_array = "device_" + item
            assert device_array in cutout_.arrays
            device_desc = cutout_.arrays[device_array]
            device_node = exit_state.add_access(device_array)

            exit_state.add_edge(
                device_node,
                None,
                host_node,
                None,
                dace.Memlet.from_array(device_array, device_desc),
            )

        return cutout_
