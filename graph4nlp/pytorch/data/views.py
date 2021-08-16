# Views implementations used in GraphData
from collections import namedtuple
from typing import Any, List, Union
import torch

from .utils import slice_to_list

NodeRepr = namedtuple("NodeData", ["features", "attributes"])
EdgeRepr = namedtuple("EdgeData", ["features"])


class NodeView(object):
    """
    View for graph nodes at at high level.
    """

    def __init__(self, graph: Any):
        self._graph = graph

    def __getitem__(self, node_idx: Union[int, slice]) -> NodeRepr:
        """
        Get a number of nodes and their corresponding data.

        Parameters
        ----------
        node_idx: int or slice
            The index of nodes to be accessed.

        Returns
        -------
        NodeRepr
            A collection of the corresponding data.
        """
        # consistency check
        # 1. type check
        if not (isinstance(node_idx, int) or isinstance(node_idx, slice)):
            raise TypeError("Only int and slice are supported currently.")
        # 2. boundary check
        if isinstance(node_idx, slice):
            node_idx_list = slice_to_list(node_idx, self._graph.get_node_num())
            for idx in node_idx_list:
                if not (idx < self._graph.get_node_num()):
                    raise ValueError("Node {} does not exist in the graph.".format(idx))
        else:
            if not (node_idx < self._graph.get_node_num()):
                raise ValueError("Node {} does not exist in the graph.".format(node_idx))

        return NodeRepr(
            features=NodeFeatView(self._graph, node_idx),
            attributes=NodeAttrView(self._graph, node_idx),
        )

    def __len__(self):
        return self._graph.get_node_num()

    def __call__(self):
        return list(range(self._graph.get_node_num))


class NodeFeatView(object):
    """
    View for node features which are all tensors.
    """

    def __init__(self, graph, nodes: Union[int, slice]):
        self._graph = graph
        self._nodes = nodes

    def __getitem__(self, feature_name: str) -> torch.Tensor:
        return self._graph.get_node_features(self._nodes)[feature_name]

    def __setitem__(self, feature_name: str, feature_value: torch.Tensor) -> None:
        self._graph.set_node_features(self._nodes, {feature_name: feature_value})

    def __repr__(self):
        return repr(self._graph.get_node_features(self._nodes))

    def keys(self):
        return self._graph.node_feature_names()


class NodeAttrView(object):
    """
    View for node attributes which are arbitrary objects.
    """

    def __init__(self, graph, nodes: Union[int, slice]):
        self._graph = graph
        self._nodes = nodes

    def __getitem__(self, item: Any) -> Any:
        total_attrs = self._graph.get_node_attrs(self._nodes)
        filtered_attrs = dict()
        for k, v in total_attrs.items():
            if item in v:
                filtered_attrs[k] = v[item]
        return filtered_attrs

    def __setitem__(self, key, value):
        raise NotImplementedError(
            "NodeAttrView does not support modifying node attributes."
            "To modify node attributes, please use GraphData.node_attributes"
        )

    def __repr__(self):
        return repr(self._graph.get_node_attrs(self._nodes))


class EdgeView(object):
    """
    View for edges at high level.
    """

    def __init__(self, graph):
        self._graph = graph

    def __call__(self, *args, **kwargs):
        return self._graph.get_all_edges(*args, **kwargs)

    def __getitem__(self, item):
        return EdgeRepr(features=EdgeFeatView(self._graph, item))

    def __repr__(self):
        return repr({"Edges": self._graph.get_all_edges()})


class EdgeFeatView(object):
    def __init__(self, graph, edges: List[int]):
        self._graph = graph
        self._edges = edges

    def __getitem__(self, feature_name: str):
        return self._graph.get_edge_feature(self._edges)[feature_name]

    def __setitem__(self, feature_name: str, feature_value: torch.Tensor):
        self._graph.set_edge_feature(self._edges, {feature_name: feature_value})

    def keys(self):
        return self._graph.get_edge_feature_names()


class BatchNodeFeatView(object):
    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, item: Any):
        return self._graph._get_batch_node_features(item)

    def __setitem__(self, key: Any, value: Any):
        return self._graph._set_batch_node_features(key, value)

    def __repr__(self):
        return self._graph._get_batch_node_features()


class BatchEdgeFeatView(object):
    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, item: Any):
        return self._graph._get_batch_edge_features(item)

    def __setitem__(self, key: Any, value: Any):
        return self._graph._set_batch_edge_features(key, value)

    def __repr__(self):
        return self._graph._get_batch_edge_features()
