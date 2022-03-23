from amr_graph_construction import (
    AmrGraphConstruction,
)


def test_amr():
    raw_data = (
        "We need to borrow 55% of the hammer price until we can get planning permission for restoration which will allow us to get a mortgage."
    )

    AmrGraphConstruction.static_topology(
        raw_data,
        merge_strategy="tailhead",
        edge_strategy="heterogeneous",
        verbose=1,
    )
    pass


if __name__ == "__main__":
    test_amr()
