from nltk.tokenize import word_tokenize
import torch
from torch import nn

from ...data.data import GraphData
from .base import DynamicGraphConstructionBase
from ..utils.generic_utils import normalize_adj, to_cuda
# from ..utils.constants import VERY_SMALL_NUMBER
from .utils import convert_adj_to_graph
from ...data.data import to_batch
from ..utils.vocab_utils import Vocab


class NodeEmbeddingBasedGraphConstruction(DynamicGraphConstructionBase):
    """Class for node embedding based dynamic graph construction.

    Parameters
    ----------
    word_vocab : Vocab
        The word vocabulary.
    embedding_styles : dict
        - ``single_token_item`` : specify whether the item (i.e., node or edge) contains single token or multiple tokens.
        - ``emb_strategy`` : specify the embedding construction strategy.
        - ``num_rnn_layers``: specify the number of RNN layers.
        - ``bert_model_name``: specify the BERT model name.
        - ``bert_lower_case``: specify whether to lower case the input text for BERT embeddings.
    """
    def __init__(self, word_vocab, embedding_styles, **kwargs):
        super(NodeEmbeddingBasedGraphConstruction, self).__init__(
                                                            word_vocab,
                                                            embedding_styles,
                                                            **kwargs)

    def forward(self, batch_graphdata):
        """Compute graph topology and initial node embeddings.

        Parameters
        ----------
        batch_graphdata : GraphData
            The input graph data.

        Returns
        -------
        GraphData
            The constructed batched graph.
        """
        batch_graphdata = self.embedding(batch_graphdata)
        batch_graphdata = self.topology(batch_graphdata)

        return batch_graphdata

    def topology(self, graph):
        """Compute graph topology.

        Parameters
        ----------
        graph : GraphData
            The input graph data.

        Returns
        -------
        GraphData
            The constructed graph.
        """
        node_emb = graph.batch_node_features["node_feat"]
        node_mask = (graph.batch_node_features["token_id"] != Vocab.PAD)

        raw_adj = self.compute_similarity_metric(node_emb, node_mask)
        raw_adj = self.sparsify_graph(raw_adj)
        graph_reg = self.compute_graph_regularization(raw_adj, node_emb)

        if self.sim_metric_type in ('rbf_kernel', 'weighted_cosine'):
            assert raw_adj.min().item() >= 0, 'adjacency matrix must be non-negative!'
            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=torch.finfo(torch.float32).eps)
            reverse_adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-2, keepdim=True), min=torch.finfo(torch.float32).eps)
        elif self.sim_metric_type == 'cosine':
            raw_adj = (raw_adj > 0).float()
            adj = normalize_adj(raw_adj)
            reverse_adj = adj
        else:
            adj = torch.softmax(raw_adj, dim=-1)
            reverse_adj = torch.softmax(raw_adj, dim=-2)

        graph = convert_adj_to_graph(graph, adj, reverse_adj, 0)
        graph.graph_attributes['graph_reg'] = graph_reg

        return graph

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
