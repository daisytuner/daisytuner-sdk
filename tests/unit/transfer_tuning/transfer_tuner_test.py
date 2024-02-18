import copy
import dace
import numpy as np

from daisytuner.benchmarking import CPUBenchmark, GPUBenchmark
from daisytuner.embeddings import MapNest
from daisytuner.normalization import MapExpandedForm
from daisytuner.transfer_tuning import TransferTuner


def test_vecadd():
    @dace.program
    def sdfg_vecadd(A: dace.float64[256], B: dace.float64[256], C: dace.float64[256]):
        for k in dace.map[0:256]:
            with dace.tasklet:
                a << A[k]
                b << B[k]
                c >> C[k]

                c = a + b

    sdfg = sdfg_vecadd.to_sdfg(simplify=True)

    preprocess = MapExpandedForm()
    preprocess.apply_pass(sdfg, {})

    root = None
    for node in sdfg.start_state.nodes():
        if (
            isinstance(node, dace.nodes.MapEntry)
            and sdfg.start_state.entry_node(node) is None
        ):
            root = node
            break

    map_nest = MapNest(sdfg.start_state, root)

    A = np.random.random((1024,)).astype(np.float64)
    B = np.random.random((1024,)).astype(np.float64)
    C = np.zeros((1024,), dtype=np.float64)
    C_opt = np.zeros((1024,), dtype=np.float64)
    args = {"A": A, "B": B, "C": C}

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
    tuner = TransferTuner(
        map_nest=map_nest,
        arguments=copy.deepcopy(args),
        device="cpu",
        benchmark=benchmark,
    )
    sdfg_opt, _ = tuner.tune()
    sdfg_opt.validate()

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_mxv():
    N = dace.symbol("N")
    K = dace.symbol("K")

    @dace.program
    def sdfg_mxv(A: dace.float64[N, K], B: dace.float64[K], C: dace.float64[N]):
        for i, k in dace.map[0:N, 0:K]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k]
                c >> C(1, lambda e, f: e + f)[i]

                c = a * b

    sdfg = sdfg_mxv.to_sdfg()
    sdfg.specialize({"N": 1024, "K": 128})
    sdfg.simplify()

    A = np.random.random((1024, 128)).astype(np.float64)
    B = np.random.random((128,)).astype(np.float64)
    C = np.zeros((1024,), dtype=np.float64)
    C_opt = np.zeros((1024,), dtype=np.float64)
    args = {"A": A, "B": B, "C": C}

    preprocess = MapExpandedForm()
    preprocess.apply_pass(sdfg, {})

    root = None
    for node in sdfg.start_state.nodes():
        if (
            isinstance(node, dace.nodes.MapEntry)
            and sdfg.start_state.entry_node(node) is None
        ):
            root = node
            break

    map_nest = MapNest(sdfg.start_state, root)

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
    tuner = TransferTuner(
        map_nest=map_nest,
        arguments=copy.deepcopy(args),
        device="cpu",
        benchmark=benchmark,
    )
    sdfg_opt, _ = tuner.tune()
    sdfg_opt.validate()

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_matmul():
    @dace.program
    def matmul(
        A: dace.float64[1024, 1024],
        B: dace.float64[1024, 1024],
        C: dace.float64[1024, 1024],
    ):
        for i, j, k in dace.map[0:1024, 0:1024, 0:1024]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k, j]
                c >> C(1, lambda a, b: a + b)[i, j]

                c = a * b

    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    A = np.random.random((1024, 1024)).astype(np.float64)
    B = np.random.random((1024, 1024)).astype(np.float64)
    C = np.zeros((1024, 1024), dtype=np.float64)
    C_opt = np.zeros((1024, 1024), dtype=np.float64)
    args = {"A": A, "B": B, "C": C}

    preprocess = MapExpandedForm()
    preprocess.apply_pass(sdfg, {})

    root = None
    for node in sdfg.start_state.nodes():
        if (
            isinstance(node, dace.nodes.MapEntry)
            and sdfg.start_state.entry_node(node) is None
        ):
            root = node
            break

    map_nest = MapNest(sdfg.start_state, root)

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
    tuner = TransferTuner(
        map_nest=map_nest,
        arguments=copy.deepcopy(args),
        device="cpu",
        benchmark=benchmark,
    )
    sdfg_opt, _ = tuner.tune()
    sdfg_opt.validate()

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_vecadd_gpu():
    @dace.program
    def sdfg_vecadd(A: dace.float64[256], B: dace.float64[256], C: dace.float64[256]):
        for k in dace.map[0:256]:
            with dace.tasklet:
                a << A[k]
                b << B[k]
                c >> C[k]

                c = a + b

    sdfg = sdfg_vecadd.to_sdfg(simplify=True)

    preprocess = MapExpandedForm()
    preprocess.apply_pass(sdfg, {})

    root = None
    for node in sdfg.start_state.nodes():
        if (
            isinstance(node, dace.nodes.MapEntry)
            and sdfg.start_state.entry_node(node) is None
        ):
            root = node
            break

    map_nest = MapNest(sdfg.start_state, root)

    A = np.random.random((1024,)).astype(np.float64)
    B = np.random.random((1024,)).astype(np.float64)
    C = np.zeros((1024,), dtype=np.float64)
    C_opt = np.zeros((1024,), dtype=np.float64)
    args = {"A": A, "B": B, "C": C}

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
    tuner = TransferTuner(
        map_nest=map_nest,
        arguments=copy.deepcopy(args),
        device="gpu",
        benchmark=benchmark,
    )
    sdfg_opt, _ = tuner.tune()
    sdfg_opt = TransferTuner.as_gpu_schedule(sdfg_opt)
    sdfg_opt.validate()

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_mxv_gpu():
    N = dace.symbol("N")
    K = dace.symbol("K")

    @dace.program
    def sdfg_mxv(A: dace.float64[N, K], B: dace.float64[K], C: dace.float64[N]):
        for i, k in dace.map[0:N, 0:K]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k]
                c >> C(1, lambda e, f: e + f)[i]

                c = a * b

    sdfg = sdfg_mxv.to_sdfg()
    sdfg.specialize({"N": 1024, "K": 128})
    sdfg.simplify()

    A = np.random.random((1024, 128)).astype(np.float64)
    B = np.random.random((128,)).astype(np.float64)
    C = np.zeros((1024,), dtype=np.float64)
    C_opt = np.zeros((1024,), dtype=np.float64)
    args = {"A": A, "B": B, "C": C}

    preprocess = MapExpandedForm()
    preprocess.apply_pass(sdfg, {})

    root = None
    for node in sdfg.start_state.nodes():
        if (
            isinstance(node, dace.nodes.MapEntry)
            and sdfg.start_state.entry_node(node) is None
        ):
            root = node
            break

    map_nest = MapNest(sdfg.start_state, root)

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
    tuner = TransferTuner(
        map_nest=map_nest,
        arguments=copy.deepcopy(args),
        device="gpu",
        benchmark=benchmark,
    )
    sdfg_opt, _ = tuner.tune()
    sdfg_opt = TransferTuner.as_gpu_schedule(sdfg_opt)
    sdfg_opt.validate()

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


def test_matmul_gpu():
    @dace.program
    def matmul(
        A: dace.float64[1024, 1024],
        B: dace.float64[1024, 1024],
        C: dace.float64[1024, 1024],
    ):
        for i, j, k in dace.map[0:1024, 0:1024, 0:1024]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k, j]
                c >> C(1, lambda a, b: a + b)[i, j]

                c = a * b

    sdfg = matmul.to_sdfg()
    sdfg.simplify()

    A = np.random.random((1024, 1024)).astype(np.float64)
    B = np.random.random((1024, 1024)).astype(np.float64)
    C = np.zeros((1024, 1024), dtype=np.float64)
    C_opt = np.zeros((1024, 1024), dtype=np.float64)
    args = {"A": A, "B": B, "C": C}

    preprocess = MapExpandedForm()
    preprocess.apply_pass(sdfg, {})

    root = None
    for node in sdfg.start_state.nodes():
        if (
            isinstance(node, dace.nodes.MapEntry)
            and sdfg.start_state.entry_node(node) is None
        ):
            root = node
            break

    map_nest = MapNest(sdfg.start_state, root)

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
    tuner = TransferTuner(
        map_nest=map_nest,
        arguments=copy.deepcopy(args),
        device="gpu",
        benchmark=benchmark,
    )
    sdfg_opt, _ = tuner.tune()
    sdfg_opt = TransferTuner.as_gpu_schedule(sdfg_opt)
    sdfg_opt.validate()

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)
