from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from stanfordcorenlp import StanfordCoreNLP

def test_dependency():
    raw_data = "James went to the corner-shop. He want to buy some (eggs), <milk> and bread for breakfast."

    embedding_styles = {
        'word_emb_type': 'glove',
        'node_edge_level_emb_type': 'mean',
        'graph_level_emb_type': 'identity'
    }
    app = DependencyBasedGraphConstruction(embedding_styles, None)
    nlp_parser = StanfordCoreNLP('http://localhost', port=9000, timeout=300000)
    app.topology(raw_data, nlp_parser, None, None)
    pass


if __name__ == "__main__":
    test_dependency()