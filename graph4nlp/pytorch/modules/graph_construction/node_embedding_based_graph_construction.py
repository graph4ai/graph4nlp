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

        node_emb = self.embedding(batch_gd, node_size, num_nodes)

        node_mask = self._get_node_mask_for_batch_graph(num_nodes)
        new_batch_gd = self.topology(node_emb, node_mask=node_mask)
        new_batch_gd.node_features['node_feat'] = node_emb
        # new_batch_gd.batch = batch_gd.batch
        new_batch_gd.copy_batch_info(batch_gd)

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
        """Convert raw text data to the initial node set graph (i.e., no edge information).

        Parameters
        ----------
        raw_text_data : str or list/tuple of str
            The raw text data. When a list/tuple of tokens is provided, no
            tokenization will be conducted and each token is a node;
            otherwise, tokenization will be conducted on the input string
            to get a list of tokens.
        lower_case : boolean
            Specify whether to lower case the input text, default: ``True``.
        tokenizer : callable, optional
            The tokenization function.

        Returns
        -------
        GraphData
            The constructed graph.
        """
        if isinstance(raw_text_data, str):
            token_list = tokenizer(raw_text_data.strip())
        elif isinstance(raw_text_data, (list, tuple)):
            token_list = raw_text_data
        else:
            raise RuntimeError('raw_text_data must be str or list/tuple of str')

        graph = GraphData()
        graph.add_nodes(len(token_list))

        for idx in range(len(token_list)):
            graph.node_attributes[idx]['token'] = token_list[idx].lower() if lower_case else token_list[idx]

        return graph