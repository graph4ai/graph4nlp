from collections import namedtuple

from .utils import slice_to_list

NodeRepr = namedtuple('NodeData', ['features', 'attributes'])
EdgeRepr = namedtuple('EdgeData', ['features'])


class NodeView(object):
    """
    View for graph nodes at at high level.
    """

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, item: int or slice):
        assert isinstance(item, int) or isinstance(item, slice), 'Only int or slice is supported in accessing nodes.'

        if isinstance(item, slice):
            node_idx_list = slice_to_list(item, self._graph.get_node_num())
            for node_idx in node_idx_list:
                assert node_idx < self._graph.get_node_num(), 'Node {} does not exist in the graph.'.format(node_idx)
        else:
            assert item < self._graph.get_node_num(), 'Node {} does not exist in the graph.'.format(item)

        return NodeRepr(features=NodeFeatView(self._graph, item), attributes=NodeAttrView(self._graph, item))

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

    def __repr__(self):
        return repr(self._graph.get_node_features(self._nodes))


class NodeAttrView(object):
    def __init__(self, graph, nodes):
        self._graph = graph
        self._nodes = nodes

    def __getitem__(self, item):
        total_attrs = self._graph._get_node_attrs(self._nodes)
        filtered_attrs = dict()
        for k, v in total_attrs.items():
            if item in v:
                filtered_attrs[k] = v[item]
        return filtered_attrs

    def __setitem__(self, key, value):
        raise NotImplementedError('NodeAttrView does not support modifying node attributes.'
                                  'To modify node attributes, please use GraphData.node_attributes')

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
        return repr({'Edges': self._graph.get_all_edges()})


class EdgeFeatView(object):
    def __init__(self, graph, edges):
        self._graph = graph
        self._edges = edges

    def __getitem__(self, item):
        return self._graph._get_edge_feature(self._edges)[item]

    def __setitem__(self, key, value):
        self._graph._set_edge_feature(self._edges, {key: value})
