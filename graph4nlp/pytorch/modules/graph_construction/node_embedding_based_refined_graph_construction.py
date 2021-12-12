import torch
from nltk.tokenize import word_tokenize

from ...data.data import GraphData
from ..utils.generic_utils import normalize_adj
from ..utils.vocab_utils import Vocab
from .base import DynamicGraphConstructionBase
from .constituency_graph_construction import ConstituencyBasedGraphConstruction
from .dependency_graph_construction import DependencyBasedGraphConstruction
from .ie_graph_construction import IEBasedGraphConstruction
from .utils import convert_adj_to_graph


class NodeEmbeddingBasedRefinedGraphConstruction(DynamicGraphConstructionBase):
    """Class for node embedding based refined dynamic graph construction.

    Parameters
    ----------
    alpha_fusion : float
        Specify the fusion value for combining initial and learned adjacency matrices.
    """

    def __init__(self, alpha_fusion, **kwargs):
        super(NodeEmbeddingBasedRefinedGraphConstruction, self).__init__(**kwargs)
        assert 0 <= alpha_fusion <= 1, "alpha_fusion should be a `float` number between 0 and 1"
        self.alpha_fusion = alpha_fusion

    def dynamic_topology(self, graph):
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
        node_mask = graph.batch_node_features["token_id"] != Vocab.PAD

        raw_adj = self.compute_similarity_metric(node_emb, node_mask)
        raw_adj = self.sparsify_graph(raw_adj)
        graph_reg = self.compute_graph_regularization(raw_adj, node_emb)

        if self.sim_metric_type in ("rbf_kernel", "weighted_cosine"):
            assert raw_adj.min().item() >= 0, "adjacency matrix must be non-negative!"
            adj = raw_adj / torch.clamp(
                torch.sum(raw_adj, dim=-1, keepdim=True), min=torch.finfo(torch.float32).eps
            )
            reverse_adj = raw_adj / torch.clamp(
                torch.sum(raw_adj, dim=-2, keepdim=True), min=torch.finfo(torch.float32).eps
            )
        elif self.sim_metric_type == "cosine":
            raw_adj = (raw_adj > 0).float()
            adj = normalize_adj(raw_adj)
            reverse_adj = adj
        else:
            adj = torch.softmax(raw_adj, dim=-1)
            reverse_adj = torch.softmax(raw_adj, dim=-2)

        if self.alpha_fusion is not None:
            # Compute the symmetric normalized Laplacian matrix of the input graph
            init_norm_adj = self._get_normalized_init_adj(graph)

            adj = torch.sparse.FloatTensor.add(
                (1 - self.alpha_fusion) * adj, self.alpha_fusion * init_norm_adj
            )
            reverse_adj = torch.sparse.FloatTensor.add(
                (1 - self.alpha_fusion) * reverse_adj, self.alpha_fusion * init_norm_adj
            )

        graph = convert_adj_to_graph(graph, adj, reverse_adj, 0)
        graph.graph_attributes["graph_reg"] = graph_reg

        return graph

    @classmethod
    def init_topology(
        cls,
        raw_text_data,
        lower_case=True,
        tokenizer=word_tokenize,
        nlp_processor=None,
        processor_args=None,
        merge_strategy=None,
        edge_strategy=None,
        verbose=False,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
    ):
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
            Strategy to merge sub-graphs into one graph, depends on specific
            ``dynamic_init_topology_builder``, default: ``None``.
        edge_strategy: str
            Strategy to process edge, depends on specific ``dynamic_init_topology_builder``,
            default: ``None``.
        verbose: boolean
            verbose flag, default: ``False``.
        dynamic_init_topology_builder : class, optional
            The initial graph topology builder, default: ``None``.
        dynamic_init_topology_aux_args : dict, optional
            The auxiliary args for dynamic_init_topology_builder.topology, default: ``None``.

        Returns
        -------
        GraphData
            The constructed graph.
        """
        if dynamic_init_topology_builder is None:  # line graph
            if isinstance(raw_text_data, str):
                token_list = tokenizer(raw_text_data.strip())
            elif isinstance(raw_text_data, (list, tuple)):
                token_list = raw_text_data
            else:
                raise RuntimeError("raw_text_data must be str or list/tuple of str")

            graph = GraphData()
            graph.add_nodes(len(token_list))

            for idx in range(len(token_list) - 1):
                graph.add_edge(idx, idx + 1)
                graph.add_edge(idx + 1, idx)
                graph.node_attributes[idx]["token"] = (
                    token_list[idx].lower() if lower_case else token_list[idx]
                )

            graph.node_attributes[idx + 1]["token"] = (
                token_list[-1].lower() if lower_case else token_list[-1]
            )
        elif dynamic_init_topology_builder in (
            IEBasedGraphConstruction,
            DependencyBasedGraphConstruction,
            ConstituencyBasedGraphConstruction,
        ):
            graph = dynamic_init_topology_builder.topology(
                raw_text_data=raw_text_data,
                nlp_processor=nlp_processor,
                processor_args=processor_args,
                merge_strategy=merge_strategy,
                edge_strategy=edge_strategy,
                verbose=verbose,
            )
        else:
            graph = dynamic_init_topology_builder.topology(
                raw_text_data, dynamic_init_topology_aux_args
            )

        return graph
