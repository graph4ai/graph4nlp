from amr_graph_construction import (
    AmrGraphConstruction,
)


def test_amr():
    raw_data = (
        "find all languageid0 job in locid0"
    )

    AmrGraphConstruction.static_topology(
        raw_data,
        verbose=1,
    )
    pass


if __name__ == "__main__":
    test_amr()
