from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from stanfordcorenlp import StanfordCoreNLP
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel


def test_dependency():
    raw_data = "James went to the corner-shop. He want to buy some (eggs), <milk> and bread for breakfast."

    embedding_styles = {
        'word_emb_type': 'glove',
        'node_edge_level_emb_type': 'mean',
        'graph_level_emb_type': 'identity'
    }
    nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=300000)
    raw_text_data = ["James went to the corner-shop."], ["He want to buy some (eggs), <milk> and bread for breakfast."]
    vocab_model = VocabModel(raw_text_data, max_word_vocab_size=None,
                             min_word_vocab_freq=1,
                             word_emb_size=300)
    app = DependencyBasedGraphConstruction(embedding_style=embedding_styles, vocab=vocab_model)

    app.topology(raw_data, nlp_parser, merge_strategy="sequential", edge_strategy=None)
    pass


if __name__ == "__main__":
    test_dependency()