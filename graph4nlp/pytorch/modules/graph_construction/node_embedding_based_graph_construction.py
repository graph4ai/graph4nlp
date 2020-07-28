import torch
from torch import nn

from .base import DynamicGraphConstructionBase
from .utils import convert_adj_to_dgl_graph


class NodeEmbeddingBasedGraphConstruction(DynamicGraphConstructionBase):
    """Class for node embedding based dynamic graph construction.

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
    def __init__(self, word_vocab, embedding_styles, **kwargs):
        super(NodeEmbeddingBasedGraphConstruction, self).__init__(
                                                            word_vocab,
                                                            embedding_styles,
                                                            **kwargs)

    def forward(self, node_word_idx, node_size, num_nodes, node_mask=None):
        """Compute graph topology and initial node/edge embeddings.

        Parameters
        ----------
        """
        node_emb = self.embedding(node_word_idx, node_size, num_nodes)

        dgl_graph = self.topology(node_emb, node_mask)
        dgl_graph.ndata['node_feat'] = node_emb

        return dgl_graph

    def topology(self, node_emb, node_mask=None):

        """Compute graph topology.

        Parameters
        ----------
        node_emb : torch.Tensor
            The node embeddings.
        node_mask : torch.Tensor, optional
            The node mask matrix, default: ``None``.

        """
        adj = self.compute_similarity_metric(node_emb, node_mask)
        dgl_graph = convert_adj_to_dgl_graph(adj, self.mask_off_val, use_edge_softmax=True)

        return dgl_graph


    def embedding(self, node_word_idx, node_size, num_nodes):
        """Compute initial node embeddings.

        Parameters
        ----------
        """
        return self.embedding_layer(node_word_idx, node_size, num_nodes)
