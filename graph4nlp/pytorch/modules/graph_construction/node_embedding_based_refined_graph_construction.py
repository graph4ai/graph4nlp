import torch
from torch import nn

from .base import DynamicGraphConstructionBase
from .utils import convert_adj_to_dgl_graph
from ..utils.generic_utils import normalize_adj
from ..utils.constants import VERY_SMALL_NUMBER


class NodeEmbeddingBasedRefinedGraphConstruction(DynamicGraphConstructionBase):
    """Class for node embedding based refined dynamic graph construction.

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
    alpha_fusion : float
        Specify the fusion value for combining initial and learned adjacency matrices.
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
    def __init__(self, word_vocab, embedding_styles, alpha_fusion, **kwargs):
        super(NodeEmbeddingBasedRefinedGraphConstruction, self).__init__(
                                                            word_vocab,
                                                            embedding_styles,
                                                            **kwargs)
        assert 0 <= alpha_fusion <= 1, 'alpha_fusion should be a `float` number between 0 and 1'
        self.alpha_fusion = alpha_fusion

    def forward(self, init_norm_adj, node_word_idx, node_size, num_nodes, node_mask=None):
        """Compute graph topology and initial node embeddings.

        Parameters
        ----------
        init_norm_adj : torch.sparse.FloatTensor
            The initial normalized adjacency matrix.
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

        dgl_graph = self.topology(node_emb, init_norm_adj, node_mask)
        dgl_graph.ndata['node_feat'] = node_emb

        return dgl_graph

    def topology(self, node_emb, init_norm_adj, node_mask=None):
        """Compute graph topology.

        Parameters
        ----------
        node_emb : torch.Tensor
            The node embeddings.
        init_norm_adj : torch.sparse.FloatTensor
            The initial init_norm_adj adjacency matrix.
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

        if self.sim_metric_type in ('rbf_kernel', 'weighted_cosine'):
            try:
                assert adj.min().item() >= 0, 'adjacency matrix must be non-negative!'
            except:
                import pdb;pdb.set_trace()
            adj = adj / torch.clamp(torch.sum(adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        elif self.sim_metric_type == 'cosine':
            adj = (adj > 0).float()
            adj = normalize_adj(adj)
        else:
            adj = torch.softmax(adj, dim=-1)

        if self.alpha_fusion is not None:
            adj = torch.sparse.FloatTensor.add((1 - self.alpha_fusion) * adj, self.alpha_fusion * init_norm_adj)

        dgl_graph = convert_adj_to_dgl_graph(adj, 0, use_edge_softmax=False)
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
