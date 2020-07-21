from torch import nn

from .embedding_construction import EmbeddingConstruction

class GraphConstructionBase(nn.Module):
    """
    Base class for graph construction.

    ...

    Attributes
    ----------
    embedding_styles : dict
        Specify embedding styles including ``word_emb_type``, ``node_edge_level_emb_type`` and ``graph_level_emb_type``.

    Methods
    -------
    forward(raw_text_data)
        Generate graph topology and embeddings.

    topology()
        Generate graph topology.

    embedding()
        Generate graph embeddings.
    """

    def __init__(self, word_vocab, embedding_styles, hidden_size,
                        fix_word_emb=True, dropout=None, use_cuda=True):
        super(GraphConstructionBase, self).__init__()
        self.embedding_layer = EmbeddingConstruction(word_vocab,
                                        embedding_styles['word_emb_type'],
                                        embedding_styles['node_edge_level_emb_type'],
                                        embedding_styles['graph_level_emb_type'],
                                        hidden_size,
                                        fix_word_emb=fix_word_emb,
                                        dropout=dropout,
                                        use_cuda=use_cuda)


    def forward(self, raw_text_data):
        raise NotImplementedError()

    def topology(self):
        raise NotImplementedError()

    def embedding(self):
        raise NotImplementedError()

class StaticGraphConstructionBase(GraphConstructionBase):
    """
    Base class for static graph construction.

    ...

    Attributes
    ----------
    embedding_styles : dict
        Specify embedding styles including ``word_emb_type``, ``node_edge_level_emb_type`` and ``graph_level_emb_type``.

    Methods
    -------
    add_vocab()
        Add new parsed words or syntactic components into vocab.

    topology()
        Generate graph topology.

    embedding(raw_data, structure)
        Generate graph embeddings.

    forward(raw_data)
        Generate static graph embeddings and topology.
    """

    def __init__(self, word_vocab, embedding_styles, hidden_size,
                 fix_word_emb=True, dropout=None, use_cuda=True):
        super(StaticGraphConstructionBase, self).__init__(word_vocab,
                                                           embedding_styles,
                                                           hidden_size,
                                                           fix_word_emb=fix_word_emb,
                                                           dropout=dropout,
                                                           use_cuda=use_cuda)

    def add_vocab(self, **kwargs):
        raise NotImplementedError()

    def topology(self, **kwargs):
        raise NotImplementedError()

    def embedding(self, **kwargs):
        raise NotImplementedError()

    def forward(self, **kwargs):
        raise NotImplementedError()

    def _construct_static_graph(self, **kwargs):
        raise NotImplementedError()

    def _graph_connect(self, **kwargs):
        raise NotImplementedError()

class DynamicGraphConstructionBase(GraphConstructionBase):
    """
    Base class for dynamic graph construction.

    ...

    Attributes
    ----------
    embedding_styles : dict
        Specify ``word_emb_type``, ``node_edge_level_emb_type`` and ``graph_level_emb_type``.

    Methods
    -------
    forward(raw_text_data)
        Generate dynamic graph topology and embeddings.

    topology()
        Generate dynamic graph topology.

    embedding(feat)
        Generate dynamic graph embeddings.
    """

    def __init__(self, word_vocab, embedding_styles, hidden_size,
                        fix_word_emb=True, dropout=None, use_cuda=True):
        super(DynamicGraphConstructionBase, self).__init__(word_vocab,
                                                            embedding_styles,
                                                            hidden_size,
                                                            fix_word_emb=fix_word_emb,
                                                            dropout=dropout,
                                                            use_cuda=use_cuda)

    def forward(self, raw_text_data):
        raise NotImplementedError()

    def topology(self, node_emb, edge_emb=None, init_adj=None, node_mask=None):
        raise NotImplementedError()

    def embedding(self, feat):
        raise NotImplementedError()
