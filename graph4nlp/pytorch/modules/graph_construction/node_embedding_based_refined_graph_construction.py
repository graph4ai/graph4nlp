from nltk.tokenize import word_tokenize
import torch
from torch import nn

from ...data.data import GraphData
from .base import DynamicGraphConstructionBase
from .dependency_graph_construction import DependencyBasedGraphConstruction
from .constituency_graph_construction import ConstituencyBasedGraphConstruction
from .ie_graph_construction import IEBasedGraphConstruction
from ..utils.generic_utils import normalize_adj, to_cuda
# from ..utils.constants import VERY_SMALL_NUMBER
from .utils import convert_adj_to_graph
from ...data.data import to_batch
from ..utils.vocab_utils import Vocab


class NodeEmbeddingBasedRefinedGraphConstruction(DynamicGraphConstructionBase):
    """Class for node embedding based refined dynamic graph construction.

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
    alpha_fusion : float
        Specify the fusion value for combining initial and learned adjacency matrices.
    """
    def __init__(self,
                word_vocab,
                embedding_styles,
                alpha_fusion,
                **kwargs):
        super(NodeEmbeddingBasedRefinedGraphConstruction, self).__init__(
                                                            word_vocab,
                                                            embedding_styles,
                                                            **kwargs)
        assert 0 <= alpha_fusion <= 1, 'alpha_fusion should be a `float` number between 0 and 1'
        self.alpha_fusion = alpha_fusion

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
        init_norm_adj = self._get_normalized_init_adj(batch_graphdata)
        batch_graphdata = self.embedding(batch_graphdata)
        batch_graphdata = self.topology(batch_graphdata, init_norm_adj)

        return batch_graphdata


    def topology(self, graph, init_norm_adj):
        """Compute graph topology.

        Parameters
        ----------
        graph : GraphData
            The input graph data.
        init_norm_adj : torch.sparse.FloatTensor
            The initial init_norm_adj adjacency matrix.

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

        if self.alpha_fusion is not None:
            adj = torch.sparse.FloatTensor.add((1 - self.alpha_fusion) * adj, self.alpha_fusion * init_norm_adj)
            reverse_adj = torch.sparse.FloatTensor.add((1 - self.alpha_fusion) * reverse_adj, self.alpha_fusion * init_norm_adj)

        graph = convert_adj_to_graph(graph, adj, reverse_adj, 0)
        graph.graph_attributes['graph_reg'] = graph_reg

        return graph

    @classmethod
    def init_topology(cls,
                    raw_text_data,
                    lower_case=True,
                    tokenizer=word_tokenize,
                    nlp_processor=None,
                    processor_args=None,
                    merge_strategy=None,
                    edge_strategy=None,
                    verbase=False,
                    dynamic_init_topology_builder=None,
                    dynamic_init_topology_aux_args=None):
        """Convert raw text data to the initial graph.

        Parameters
        ----------
        raw_text_data : str or list/tuple of str
            The raw text data. When a list/tuple of tokens is provided, no
            tokenization will be conducted and each token is a node
            (used for line graph builder); otherwise, tokenization will
            be conducted on the input string to get a list of tokens.
        lower_case : boolean
            Specify whether to lower case the input text, default: ``True``.
        tokenizer : callable, optional
            The tokenization function, default: ``nltk.tokenize.word_tokenize``.
        nlp_processor: StanfordCoreNLP, optional
            The NLP processor, default: ``None``.
        processor_args: dict, optional
            The NLP processor arguments, default: ``None``.
        merge_strategy: str
            Strategy to merge sub-graphs into one graph, depends on specific ``dynamic_init_topology_builder``, default: ``None``.
        edge_strategy: str
            Strategy to process edge, depends on specific ``dynamic_init_topology_builder``, default: ``None``.
        verbase: boolean
            verbase flag, default: ``False``.
        dynamic_init_topology_builder : class, optional
            The initial graph topology builder, default: ``None``.
        dynamic_init_topology_aux_args : dict, optional
            The auxiliary args for dynamic_init_topology_builder.topology, default: ``None``.

        Returns
        -------
        GraphData
            The constructed graph.
        """
        if dynamic_init_topology_builder is None: # line graph
            if isinstance(raw_text_data, str):
                token_list = tokenizer(raw_text_data.strip())
            elif isinstance(raw_text_data, (list, tuple)):
                token_list = raw_text_data
            else:
                raise RuntimeError('raw_text_data must be str or list/tuple of str')

            graph = GraphData()
            graph.add_nodes(len(token_list))

            for idx in range(len(token_list) - 1):
                graph.add_edge(idx, idx + 1)
                graph.add_edge(idx + 1, idx)
                graph.node_attributes[idx]['token'] = token_list[idx].lower() if lower_case else token_list[idx]

            graph.node_attributes[idx + 1]['token'] = token_list[-1].lower() if lower_case else token_list[-1]
        elif dynamic_init_topology_builder in (IEBasedGraphConstruction, DependencyBasedGraphConstruction, ConstituencyBasedGraphConstruction):
            graph = dynamic_init_topology_builder.topology(
                                raw_text_data=raw_text_data,
                                nlp_processor=nlp_processor,
                                processor_args=processor_args,
                                merge_strategy=merge_strategy,
                                edge_strategy=edge_strategy,
                                verbase=verbase)
        else:
            graph = dynamic_init_topology_builder.topology(
                                raw_text_data,
                                dynamic_init_topology_aux_args)

        return graph
