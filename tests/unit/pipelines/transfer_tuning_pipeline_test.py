import dace
import numpy as np

from daisytuner.benchmarking import CPUBenchmark
from daisytuner.normalization import MapExpandedForm
from daisytuner.pipelines import TransferTuningPipeline


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
    sdfg(A=A, B=B, C=C)

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

    preprocess = MapExpandedForm()
    preprocess.apply_pass(sdfg, {})

    res = {}
    pipeline = TransferTuningPipeline(benchmark=benchmark, device="cpu")
    pipeline.apply_pass(sdfg, res)

    C_opt = np.zeros((1024, 1024), dtype=np.float64)
    sdfg(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)
