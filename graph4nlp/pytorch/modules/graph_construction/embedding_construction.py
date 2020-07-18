from torch import nn

class EmbeddingConstructionBase(nn.Module):
    """
    Base class for (initial) graph embedding construction.

    ...

    Attributes
    ----------
    feat : dict
        Raw features of graph nodes and/or edges.

    Methods
    -------
    forward(raw_text_data)
        Generate dynamic graph topology and embeddings.
    """

    def __init__(self):
        super(EmbeddingConstructionBase, self).__init__()

    def forward(self):
        raise NotImplementedError()

class EmbeddingConstruction(EmbeddingConstructionBase):
    """
    Graph embedding construction class.

    ...

    Attributes
    ----------
    word_emb_type: list of str
        Specify pretrained word embedding types.

    node_edge_level_emb_type: str
        Specify node/edge level embedding initialization strategies (e.g., ``mean`` and ``bilstm``).

    graph_level_emb_type: str
        Specify graph level embedding initialization strategies (e.g., ``identity`` and ``bilstm``).

    fix_word_emb: boolean, default: ``True``
        Specify whether to fix pretrained word embeddings.


    Methods
    -------
    forward(feat)
        Generate initial node and/or edge embeddings for the input graph.
    """

    def __init__(self, word_emb_type, node_edge_level_emb_type, graph_level_emb_type, fix_word_emb=True):
        super(EmbeddingConstruction, self).__init__()

        self.word_embs = nn.ModuleList()
        if 'glove' in word_emb_type:
            self.word_embs.append(GloveEmbedding(fix_word_emb))

        if 'bert' in word_emb_type:
            self.word_embs.append(BertEmbedding(fix_word_emb))

        if node_edge_level_emb_type == 'mean':
            pass
            # self.node_level_emb = MeanEmbedding()
        elif node_edge_level_emb_type == 'bilstm':
            self.node_level_emb = BiLSTMEmbedding()
        else:
            raise RuntimeError('Unknown node_edge_level_emb_type: {}'.format(node_edge_level_emb_type))

        if graph_level_emb_type == 'identity':
            self.graph_level_emb = IdentityEmbedding()
        elif graph_level_emb_type == 'bilstm':
            self.graph_level_emb = BiLSTMEmbedding()
        else:
            raise RuntimeError('Unknown graph_level_emb_type: {}'.format(graph_level_emb_type))

    def forward(self, feat):
        feat = []
        for word_emb in self.word_embs:
            feat.append(word_emb(feat))
        feat = torch.stack(feat, dim=-1)

        feat = self.node_level_emb(feat)
        feat = self.graph_level_emb(feat)

        return feat

class GloveEmbedding(nn.Module):
    """
    GloVe embedding layer.

    ...

    Attributes
    ----------


    Methods
    -------
    forward(feat)
        Generate GloVe embeddings.
    """
    def __init__(self, fix_word_emb = True):
        super(GloveEmbedding, self).__init__()
        self.fix_word_emb = fix_word_emb

    def forward(self, feat):
        raise NotImplementedError()

class BertEmbedding(nn.Module):
    """
    BERT embedding layer.

    ...

    Attributes
    ----------


    Methods
    -------
    forward(feat)
        Generate BERT embeddings.
    """
    def __init__(self, fix_word_emb):
        super(BertEmbedding, self).__init__()
        self.fix_word_emb = fix_word_emb

    def forward(self, feat):
        raise NotImplementedError()

class IdentityEmbedding(nn.Module):
    """
    Identity embedding layer.

    ...

    Attributes
    ----------


    Methods
    -------
    forward(feat)
        Return the input embeddings.
    """
    def __init__(self):
        super(IdentityEmbedding, self).__init__()

    def forward(self, feat):
        raise feat

class BiLSTMEmbedding(nn.Module):
    """
    BiLSTM embedding layer.

    ...

    Attributes
    ----------


    Methods
    -------
    forward(feat)
        Apply the BiLSTM network to the input sequence of embeddings.
    """
    def __init__(self):
        super(BiLSTMEmbedding, self).__init__()

    def forward(self, feat):
        raise NotImplementedError()
