from torch import nn

from .embedding_construction import EmbeddingConstruction


class GraphConstructionBase(nn.Module):
    """Base class for graph construction.

    Parameters
    ----------
    word_vocab : Vocab
        The word vocabulary.
    embedding_styles : dict
        - ``word_emb_type`` : Specify pretrained word embedding types
            including "w2v" and/or "bert".
        - ``node_edge_emb_strategy`` : Specify node/edge embedding
            strategies including "mean", "lstm", "gru", "bilstm" and "bigru".
        - ``seq_info_encode_strategy`` : Specify strategies of encoding
            sequential information in raw text data including "none",
            "lstm", "gru", "bilstm" and "bigru".
    hidden_size : int, optional
        The hidden size of RNN layer, default: ``None``.
    fix_word_emb : boolean, optional
        Specify whether to fix pretrained word embeddings, default: ``True``.
    dropout : float, optional
        Dropout ratio, default: ``None``.
    device : torch.device, optional
        Specify computation device (e.g., CPU), default: ``None`` for using CPU.
    """
    def __init__(self, word_vocab, embedding_styles, hidden_size=None,
                        fix_word_emb=True, dropout=None, device=None):
        super(GraphConstructionBase, self).__init__()
        self.embedding_layer = EmbeddingConstruction(word_vocab,
                                        embedding_styles['word_emb_type'],
                                        embedding_styles['node_edge_emb_strategy'],
                                        embedding_styles['seq_info_encode_strategy'],
                                        hidden_size=hidden_size,
                                        fix_word_emb=fix_word_emb,
                                        dropout=dropout,
                                        device=device)

    def forward(self, raw_text_data, **kwargs):
        """Compute graph topology and initial node/edge embeddings.

        Parameters
        ----------
        raw_text_data :
            The raw text data.
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

    def topology(self, **kwargs):
        """Compute graph topology.

        Parameters
        ----------
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

    def embedding(self, **kwargs):
        """Compute initial node/edge embeddings.

        Parameters
        ----------
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

class StaticGraphConstructionBase(GraphConstructionBase):
    """
    Base class for static graph construction.

    ...

    Attributes
    ----------
    embedding_styles : dict
        Specify embedding styles including ``word_emb_type``, ``node_edge_emb_strategy`` and ``seq_info_encode_strategy``.

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
    def __init__(self, embedding_styles):
        super(StaticGraphConstructionBase, self).__init__(embedding_styles)

    def add_vocab(self):
        raise NotImplementedError()

    def topology(self, raw_text_data, nlp_processor, merge_strategy, edge_strategy):
        raise NotImplementedError()

    def embedding(self, raw_data, structure):
        raise NotImplementedError()

    def forward(self, raw_data):
        raise NotImplementedError()

class DynamicGraphConstructionBase(GraphConstructionBase):
    """Base class for dynamic graph construction.

    Parameters
    ----------
    word_vocab : Vocab
        The word vocabulary.
    embedding_styles : dict
        - ``word_emb_type`` : Specify pretrained word embedding types
            including "w2v" and/or "bert".
        - ``node_edge_emb_strategy`` : Specify node/edge embedding
            strategies including "mean", "lstm", "gru", "bilstm" and "bigru".
        - ``seq_info_encode_strategy`` : Specify strategies of encoding
            sequential information in raw text data including "none",
            "lstm", "gru", "bilstm" and "bigru".
    hidden_size : int, optional
        The hidden size of RNN layer, default: ``None``.
    fix_word_emb : boolean, optional
        Specify whether to fix pretrained word embeddings, default: ``True``.
    dropout : float, optional
        Dropout ratio, default: ``None``.
    device : torch.device, optional
        Specify computation device (e.g., CPU), default: ``None`` for using CPU.
    """

    def __init__(self, word_vocab, embedding_styles, hidden_size=None,
                        fix_word_emb=True, dropout=None, device=None):
        super(DynamicGraphConstructionBase, self).__init__(word_vocab,
                                                            embedding_styles,
                                                            hidden_size=hidden_size,
                                                            fix_word_emb=fix_word_emb,
                                                            dropout=dropout,
                                                            device=device)

    def forward(self, raw_text_data, **kwargs):
        """Compute graph topology and initial node/edge embeddings.

        Parameters
        ----------
        raw_text_data : list of sequences.
            The raw text data.
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

    def topology(self, node_emb, edge_emb=None,
                    init_adj=None, node_mask=None, **kwargs):
        """Compute graph topology.

        Parameters
        ----------
        node_emb : torch.Tensor
            The node embeddings.
        edge_emb : torch.Tensor, optional
            The edge embeddings, default: ``None``.
        init_adj : torch.Tensor, optional
            The initial adjacency matrix, default: ``None``.
        node_mask : torch.Tensor, optional
            The node mask matrix, default: ``None``.
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

    def embedding(self, feat, **kwargs):
        """Compute initial node/edge embeddings.

        Parameters
        ----------
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

    def raw_text_to_init_graph(self, raw_text_data, **kwargs):
        """Convert raw text data to initial static graph.

        Parameters
        ----------
        raw_text_data : list of sequences.
            The raw text data.
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()
