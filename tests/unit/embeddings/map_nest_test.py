import dace

from collections import Counter

from daisytuner.embeddings import MapNest


def test_valid():
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

    assert map_nest.root == root
    assert len(map_nest.nodes()) == 7
    node_statistics = Counter([type(node).__name__ for node in map_nest.nodes()])
    assert node_statistics["AccessNode"] == 2
    assert node_statistics["MapEntry"] == 2
    assert node_statistics["MapExit"] == 2
    assert node_statistics["Tasklet"] == 1


def test_invalid():
    @dace.program
    def sdfg_invalid(A: dace.float32[32, 32], B: dace.float32[32, 32]):
        for i in dace.map[0:32]:
            for j in dace.map[0:32]:
                with dace.tasklet:
                    a << A[i, j]
                    b >> B[j, i]

                    b = a + 1

    sdfg = sdfg_invalid.to_sdfg()

    access_node = None
    nested_map_entry = None
    for node in sdfg.start_state.nodes():
        if (
            isinstance(node, dace.nodes.MapEntry)
            and sdfg.start_state.entry_node(node) is not None
        ):
            nested_map_entry = node
        if isinstance(node, dace.nodes.AccessNode):
            access_node = node

    try:
        _ = MapNest(sdfg.start_state, access_node)
        assert False
    except AssertionError:
        assert True

    try:
        _ = MapNest(sdfg.start_state, nested_map_entry)
        assert False
    except AssertionError:
        assert True
