import dace
import pytest
import numpy as np

from daisytuner.benchmarking import CPUBenchmark, GPUBenchmark
from daisytuner.normalization import MapExpandedForm
from daisytuner.device_mapping import DeviceMapper


def test_cpu_schedule():
    @dace.program
    def sdfg_loop(
        A: dace.float64[512, 256], B: dace.float64[256], C: dace.float64[512]
    ):
        for i in dace.map[0:512]:
            with dace.tasklet:
                c >> C[i]
                c = 0

        for k in range(256):
            for i, j in dace.map[0:512, 0:k]:
                with dace.tasklet:
                    a << A[i, j]
                    b << B[j]
                    c >> C(1, lambda e, f: e + f)[i]

                    c = a * b

    sdfg = sdfg_loop.to_sdfg()
    sdfg.simplify()

    pipeline = MapExpandedForm()
    pipeline.apply_pass(sdfg, {})

    host_benchmark = CPUBenchmark(
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
    device_benchmark = GPUBenchmark(
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

    mapper = DeviceMapper(
        sdfg=sdfg,
        agent_type="cpu",
        cpu_benchmark=host_benchmark,
        gpu_benchmark=device_benchmark,
    )
    sdfg_opt = mapper.tune()

    A = np.random.random((512, 256)).astype(np.float64)
    B = np.random.random((256,)).astype(np.float64)
    C = np.zeros((512,), dtype=np.float64)
    C_opt = np.zeros((512,), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


@pytest.mark.skip
def test_gpu_schedule():
    @dace.program
    def sdfg_loop(
        A: dace.float64[512, 256], B: dace.float64[256], C: dace.float64[512]
    ):
        for i in dace.map[0:512]:
            with dace.tasklet:
                c >> C[i]
                c = 0

        for k in range(256):
            for i, j in dace.map[0:512, 0:k]:
                with dace.tasklet:
                    a << A[i, j]
                    b << B[j]
                    c >> C(1, lambda e, f: e + f)[i]

                    c = a * b

    sdfg = sdfg_loop.to_sdfg()
    sdfg.simplify()

    pipeline = MapExpandedForm()
    pipeline.apply_pass(sdfg, {})

    host_benchmark = CPUBenchmark(
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
    device_benchmark = GPUBenchmark(
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

    mapper = DeviceMapper(
        sdfg=sdfg,
        agent_type="gpu",
        cpu_benchmark=host_benchmark,
        gpu_benchmark=device_benchmark,
    )
    sdfg_opt = mapper.tune()

    A = np.random.random((512, 256)).astype(np.float64)
    B = np.random.random((256,)).astype(np.float64)
    C = np.zeros((512,), dtype=np.float64)
    C_opt = np.zeros((512,), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)


@pytest.mark.skip
def test_greedy_schedule():
    @dace.program
    def sdfg_loop(
        A: dace.float64[512, 256], B: dace.float64[256], C: dace.float64[512]
    ):
        for i in dace.map[0:512]:
            with dace.tasklet:
                c >> C[i]
                c = 0

        for k in range(256):
            for i, j in dace.map[0:512, 0:k]:
                with dace.tasklet:
                    a << A[i, j]
                    b << B[j]
                    c >> C(1, lambda e, f: e + f)[i]

                    c = a * b

    sdfg = sdfg_loop.to_sdfg()
    sdfg.simplify()

    pipeline = MapExpandedForm()
    pipeline.apply_pass(sdfg, {})

    host_benchmark = CPUBenchmark(
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
    device_benchmark = GPUBenchmark(
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

    mapper = DeviceMapper(
        sdfg=sdfg,
        agent_type="greedy",
        cpu_benchmark=host_benchmark,
        gpu_benchmark=device_benchmark,
    )
    sdfg_opt = mapper.tune()

    A = np.random.random((512, 256)).astype(np.float64)
    B = np.random.random((256,)).astype(np.float64)
    C = np.zeros((512,), dtype=np.float64)
    C_opt = np.zeros((512,), dtype=np.float64)

    sdfg(A=A, B=B, C=C)
    sdfg_opt(A=A, B=B, C=C_opt)
    assert np.allclose(C, C_opt)
