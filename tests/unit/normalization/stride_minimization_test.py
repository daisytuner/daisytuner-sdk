import dace

from dace.sdfg.nodes import MapEntry
from dace.transformation import pass_pipeline as ppl

from daisytuner.normalization import StrideMinimization

BSize = dace.symbol("BSize")
N, M, K = [dace.symbol(k) for k in "NMK"]


def test_matmul():

    N = 32
    M = 32
    K = 128

    @dace.program
    def matmul_collapsed(
        A: dace.float64[N, K],
        B: dace.float64[K, M],
        C: dace.float64[N, M],
    ):
        for i, j, k in dace.map[0:N, 0:M, 0:K]:
            with dace.tasklet:
                a << A[i, k]
                b << B[k, j]
                c >> C(1, lambda a, b: a + b)[i, j]

                c = a * b

    sdfg = matmul_collapsed.to_sdfg()

    res = {}
    pipeline = StrideMinimization()
    pipeline.apply_pass(sdfg, res)

    map_entry = None
    for n in sdfg.nodes()[0].nodes():
        if isinstance(n, MapEntry):
            map_entry = n
            break
    assert map_entry is not None
    assert map_entry.map.params == ["i", "k", "j"]
