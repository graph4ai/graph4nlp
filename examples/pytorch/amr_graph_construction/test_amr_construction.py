from amr_graph_construction import (
    AMRGraphConstruction,
)


def test_amr():
    raw_data = (
        "find all languageid0 job in locid0"
    )

    AMRGraphConstruction.static_topology(
        raw_data,
        verbose=1,
    )
    pass


if __name__ == "__main__":
    test_amr()
