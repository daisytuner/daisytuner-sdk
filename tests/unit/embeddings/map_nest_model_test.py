import dace

from daisytuner.benchmarking import CPUBenchmark, GPUBenchmark
from daisytuner.embeddings import MapNest, MapNestModel


def test_model_cpu():
    @dace.program
    def sdfg_valid(A: dace.float32[32, 32], B: dace.float32[32, 32]):
        for i in dace.map[0:32]:
            for j in dace.map[0:32]:
                with dace.tasklet:
                    a << A[i, j]
                    b >> B[j, i]

                    b = a + 1

    sdfg = sdfg_valid.to_sdfg()

    root = None
    for node in sdfg.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            root = node
            break

    map_nest = MapNest(sdfg.start_state, root)
    cutout = map_nest.as_cutout()

    benchmark = CPUBenchmark(
        {
            "arch": "haswellEP",
            "num_sockets": 1,
            "cores_per_socket": 12,
            "threads_per_core": 2,
            "l2_cache": 256,
            "l3_cache": 30720,
            "peakflops": 64014.11,
            "peakflops_avx": 505581.94,
            "stream_load": 54884.88,
            "stream_store": 23490.74,
            "stream_copy": 31800.04,
            "stream_triad": 35729.02,
        }
    )
    model = MapNestModel("cpu", benchmark)
    result = model.predict(cutout)
    assert result["runtime"] > 0
    assert len(result["embedding"]) == 256
    assert set(result["node_embeddings"].keys()) == set(
        [str(sdfg.start_state.node_id(node)) for node in sdfg.start_state.nodes()]
    )


def test_model_gpu():
    @dace.program
    def sdfg_valid(A: dace.float32[32, 32], B: dace.float32[32, 32]):
        for i in dace.map[0:32]:
            for j in dace.map[0:32]:
                with dace.tasklet:
                    a << A[i, j]
                    b >> B[j, i]

                    b = a + 1

    sdfg = sdfg_valid.to_sdfg()

    root = None
    for node in sdfg.start_state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            root = node
            break

    map_nest = MapNest(sdfg.start_state, root)
    cutout = map_nest.as_cutout()

    benchmark = GPUBenchmark(
        {
            "devices": 1,
            "arch": "nvidia_cc_ge_7",
            "compute_capability": 8.9,
            "l2_cache": 48,
            "memory": 11730,
            "SIMD_width": 32,
            "clock_rate": 2640,
            "mem_clock_rate": 10501,
        }
    )
    model = MapNestModel("gpu", benchmark)
    result = model.predict(cutout)
    assert result["runtime"] > 0
    assert len(result["embedding"]) == 256
    assert set(result["node_embeddings"].keys()) == set(
        [str(sdfg.start_state.node_id(node)) for node in sdfg.start_state.nodes()]
    )
