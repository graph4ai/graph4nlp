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
    sim_metric_type : str, optional
        Specify similarity metric function type including "attention",
        "weighted_cosine", "gat_attention", "rbf_kernel", and "cosine".
        Default: ``"weighted_cosine"``.
    num_heads : int, optional
        Specify the number of heads for multi-head similarity metric
        function, default: ``1``.
    top_k_neigh : int, optional
        Specify the top k value for knn neighborhood graph sparsificaiton,
        default: ``None``.
    epsilon_neigh : float, optional
        Specify the epsilon value (i.e., between ``0`` and ``1``) for
        epsilon neighborhood graph sparsificaiton, default: ``None``.
    smoothness_ratio : float, optional
        Specify the smoothness ratio (i.e., between ``0`` and ``1``)
        for graph regularization on smoothness, default: ``None``.
    connectivity_ratio : float, optional
        Specify the connectivity ratio (i.e., between ``0`` and ``1``)
        for graph regularization on connectivity, default: ``None``.
    sparsity_ratio : float, optional
        Specify the sparsity ratio (i.e., between ``0`` and ``1``)
        for graph regularization on sparsity, default: ``None``.
    input_size : int, optional
        The dimension of input embeddings, default: ``None``.
    hidden_size : int, optional
        The dimension of hidden layers, default: ``None``.
    fix_word_emb : boolean, optional
        Specify whether to fix pretrained word embeddings, default: ``False``.
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
        """Compute graph topology and initial node embeddings.

        Parameters
        ----------
        node_word_idx : torch.LongTensor
            The input word index node features.
        node_size : torch.LongTensor
            Indicate the length of word sequences for nodes.
        num_nodes : torch.LongTensor
            Indicate the number of nodes.
        node_mask : torch.Tensor, optional
            The node mask matrix, default: ``None``.

        Returns
        -------
        GraphData
            The constructed graph.
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

        Returns
        -------
        GraphData
            The constructed graph.
        """
        adj = self.compute_similarity_metric(node_emb, node_mask)
        adj = self.sparsify_graph(adj)
        graph_reg = self.compute_graph_regularization(adj, node_emb)

        dgl_graph = convert_adj_to_dgl_graph(adj, self.mask_off_val, use_edge_softmax=True)
        dgl_graph.graph_reg = graph_reg

        return dgl_graph


    def embedding(self, node_word_idx, node_size, num_nodes):
        """Compute initial node embeddings.

        Parameters
        ----------
        node_word_idx : torch.LongTensor
            The input word index node features.
        node_size : torch.LongTensor
            Indicate the length of word sequences for nodes.
        num_nodes : torch.LongTensor
            Indicate the number of nodes.

        Returns
        -------
        torch.Tensor
            The initial node embeddings.
        """
        return self.embedding_layer(node_word_idx, node_size, num_nodes)
