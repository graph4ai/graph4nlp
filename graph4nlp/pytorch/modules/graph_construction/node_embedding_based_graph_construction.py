from nltk.tokenize import word_tokenize
import torch
from torch import nn

from ...data.data import GraphData
from .base import DynamicGraphConstructionBase
from ..utils.generic_utils import normalize_adj, to_cuda
from ..utils.constants import VERY_SMALL_NUMBER
from .utils import convert_adj_to_graph
from ...data.data import to_batch


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

    def forward(self, batch_graphdata: list):
        """Compute graph topology and initial node embeddings.

        Parameters
        ----------
        batch_graphdata : list of GraphData
            The input graph data list.
        Returns
        -------
        GraphData
            The constructed batched graph.
        """
        node_size = []
        num_nodes = []

        for g in batch_graphdata:
            g.node_features['token_id'] = to_cuda(g.node_features['token_id'], self.device)
            num_nodes.append(g.get_node_num())
            node_size.extend([1 for i in range(num_nodes[-1])])

        node_size = to_cuda(torch.Tensor(node_size), self.device).int()
        num_nodes = to_cuda(torch.Tensor(num_nodes), self.device).int()
        batch_gd = to_batch(batch_graphdata)

        node_emb = self.embedding(batch_gd.node_features['token_id'].long(), node_size, num_nodes)

        node_mask = self._get_node_mask_for_batch_graph(num_nodes)
        new_batch_gd = self.topology(node_emb, node_mask=node_mask)
        new_batch_gd.node_features['node_feat'] = node_emb
        new_batch_gd.batch = batch_gd.batch

        return new_batch_gd


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
        raw_adj = self.compute_similarity_metric(node_emb, node_mask)
        raw_adj = self.sparsify_graph(raw_adj)
        graph_reg = self.compute_graph_regularization(raw_adj, node_emb)

        if self.sim_metric_type in ('rbf_kernel', 'weighted_cosine'):
            assert raw_adj.min().item() >= 0, 'adjacency matrix must be non-negative!'
            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            reverse_adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=0, keepdim=True), min=VERY_SMALL_NUMBER)
        elif self.sim_metric_type == 'cosine':
            raw_adj = (raw_adj > 0).float()
            adj = normalize_adj(raw_adj)
            reverse_adj = adj
        else:
            adj = torch.softmax(raw_adj, dim=-1)
            reverse_adj = torch.softmax(raw_adj, dim=0)

        graph_data = convert_adj_to_graph(adj, reverse_adj, 0)
        graph_data.graph_attributes['graph_reg'] = graph_reg

        return graph_data


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


    @classmethod
    def init_topology(cls, raw_text_data, lower_case=True, tokenizer=word_tokenize):
        """Convert raw text data to initial node set graph.

        Parameters
        ----------
        raw_text_data : str
            The raw text data.
        lower_case : boolean
            Specify whether to lower case the input text, default: ``True``.

        Returns
        -------
        GraphData
            The constructed graph.
        """
        if lower_case:
            raw_text_data = raw_text_data.lower()

        token_list = tokenizer(raw_text_data.strip())
        graph = GraphData()
        graph.add_nodes(len(token_list))

        for idx in range(len(token_list)):
            graph.node_attributes[idx]['token'] = token_list[idx]

        return graph