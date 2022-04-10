from amr_graph_construction import (
    AmrGraphConstruction,
)


def test_amr():
    raw_data = (
        "We need to borrow 55% of the hammer price until we can get planning permission for restoration which will allow us to get a mortgage . I saw a nice dog and noticed he was eating a bone ."
    )

    AmrGraphConstruction.static_topology(
        raw_data,
        verbose=1,
    )
    pass


if __name__ == "__main__":
    test_amr()
