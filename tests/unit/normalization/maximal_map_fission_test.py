import dace

from daisytuner.normalization import MaximalMapFission


def test_independent_writes():
    @dace.program
    def independent_writes(
        A: dace.float64[32, 32], B: dace.float64[32, 32], C: dace.float64[32, 32]
    ):
        for i, j in dace.map[0:32, 0:32]:
            a = A[i, j]
            B[i, j] = 2 * a
            C[j, i] = a + 1

    sdfg = independent_writes.to_sdfg()
    sdfg.simplify()

    pipeline = MaximalMapFission()
    pipeline.apply_pass(sdfg, {})
    assert (
        len(
            [node for node in sdfg.start_state if isinstance(node, dace.nodes.MapEntry)]
        )
        == 5
    )
