from collections import namedtuple

from graph4nlp.pytorch.data.utils import slice_to_list, SizeMismatchException

NodeRepr = namedtuple('NodeData', ['features', 'attributes'])


class NodeView(object):
    """
    View for graph nodes at at high level.
    """

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, item):
        if isinstance(item, slice):
            node_idx_list = slice_to_list(item, self._graph.get_node_num())
            for node_idx in node_idx_list:
                if node_idx >= self._graph.get_node_num():
                    raise SizeMismatchException('Node {} does not exist in the graph.'.format(node_idx))

        return NodeRepr(features=NodeFeatView(self._graph, item), attributes=self._graph.node_attributes)

    def __len__(self):
        return self._graph.get_node_num()


class NodeFeatView(object):
    """
    View for node features which are all tensors
    """

    def __init__(self, graph, nodes: slice or int):
        self._graph = graph
        self._nodes = nodes

    def __getitem__(self, item):
        return self._graph.get_node_features(self._nodes)[item]

    def __setitem__(self, key, value):
        return self._graph.set_node_features(self._nodes, {key: value})

    def __delitem__(self, key):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass

    def __repr__(self):
        return repr(self._graph.get_node_features(self._nodes))


class NodeAttrView(object):
    def __init__(self, graph, nodes):
        self._graph = graph
        self._nodes = nodes

    def __getitem__(self, item):
        return self._graph.get_node_attrs(self._nodes)[item]

    def __setitem__(self, key, value):
        pass

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
