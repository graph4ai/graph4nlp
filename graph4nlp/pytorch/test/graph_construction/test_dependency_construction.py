from stanfordcorenlp import StanfordCoreNLP

from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import (
    DependencyBasedGraphConstruction,
)


def test_dependency():
    raw_data = (
        "James went to the corner-shop. He want to buy some (eggs), <milk> and bread for breakfast."
    )

    nlp_parser = StanfordCoreNLP("http://localhost", port=9000, timeout=300000)

    processor_args = {
        "annotators": "ssplit,tokenize,depparse",
        "tokenize.options": "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",  # noqa
        "tokenize.whitespace": True,
        "ssplit.isOneSentence": True,
        "outputFormat": "json",
    }

    DependencyBasedGraphConstruction.topology(
        raw_data,
        nlp_parser,
        processor_args=processor_args,
        merge_strategy="tailhead",
        edge_strategy="heterogeneous",
        verbose=1,
    )
    pass


if __name__ == "__main__":
    test_dependency()
