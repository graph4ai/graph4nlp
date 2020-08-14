from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from stanfordcorenlp import StanfordCoreNLP
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel


def test_dependency():
    raw_data = "James went to the corner-shop. He want to buy some (eggs), <milk> and bread for breakfast."

    nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=300000)

    DependencyBasedGraphConstruction.topology(raw_data, nlp_parser, merge_strategy="tailhead",
                                              edge_strategy="heterogeneous", verbase=1)
    pass


if __name__ == "__main__":
    test_dependency()