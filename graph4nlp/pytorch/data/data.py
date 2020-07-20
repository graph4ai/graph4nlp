from collections import namedtuple

import dgl
import torch

from .utils import entail_zero_padding, slice_to_list
from .utils import SizeMismatchException, NodeNotFoundException
from .views import NodeView, NodeFeatView, EdgeView

EdgeIndex = namedtuple('EdgeIndex', ['src', 'tgt'])

node_feat_factory = dict
node_attr_factory = dict
single_node_attr_factory = dict
res_init_node_attr = {'node_attr': None}
res_init_node_features = {'node_feat': None, 'node_emb': None}


class GraphData(object):
    """
    Represent a single graph with additional attributes.
    """

    def __init__(self):
        self._node_attributes = node_attr_factory()
        self._node_features = node_feat_factory(res_init_node_features)
        self._edge_indices = EdgeIndex(src=[], tgt=[])

    @property
    def nodes(self) -> NodeView:
        """
        Return a node view through which the user can access the features and attributes

        Returns
        -------
        node: NodeView
            The node view
        """
        return NodeView(self)

    def get_node_num(self) -> int:
        """
        Get the number of nodes in the graph.

        Returns
        -------
        num_nodes: int
            The number of nodes in the graph.
        """
        return len(self._node_attributes)

    def add_nodes(self, node_num: int) -> None:
        """
        Add a number of nodes to the graph.

        Parameters
        ------
        node_num: int
            The number of nodes to be added
        """
        current_num_nodes = self.get_node_num()

        # Create placeholders in the node attribute dictionary
        for new_node_idx in range(current_num_nodes, current_num_nodes + node_num):
            self._node_attributes[new_node_idx] = single_node_attr_factory(**res_init_node_attr)

        # Do padding in the node feature dictionary
        for key in self._node_features.keys():
            entail_zero_padding(self._node_features[key], node_num)

    def get_node_features(self, nodes: int or slice) -> torch.tensor:
        """
        Get the node feature dictionary of the `nodes`

        Parameters
        ----------
        nodes: int or slice
            The nodes to be accessed

        Returns
        -------
        node_features: dict
            The reference dict of the actual tensor
        """
        ret = dict()
        for key in self._node_features.keys():
            if self._node_features[key] is None:
                ret[key] = None
            else:
                ret[key] = self._node_features[key][nodes]
        return ret

    def set_node_features(self, nodes: int or slice, new_data: dict) -> None:
        """
        Set the features of the `nodes` with the given `new_data`.

        Parameters
        ----------
        nodes: int or slice
            The nodes involved
        new_data: dict
            The new data to write. Key indicates feature name and value indicates the actual value

        Raises
        ----------
        SizeMismatch
            If the size of the new features does not match the node number

        """

        # Consistency check
        for key in new_data.keys():
            if key not in self._node_features or self._node_features[key] is None:  # A new feature is added
                # If the shape of the new feature does not match the number of existing nodes, then error occurs
                if (not isinstance(nodes, slice)) or (
                        len(slice_to_list(nodes, self.get_node_num())) != self.get_node_num()):
                    raise SizeMismatchException(
                        'The new feature should cover all existing {} nodes!'.format(self.get_node_num()))

        # Modification
        for key, value in new_data.items():
            if key not in self._node_features or self._node_features[key] is None:
                self._node_features[key] = value
            else:
                self._node_features[key][nodes] = value

    @property
    def node_attributes(self) -> dict:
        """
        Access node attribute dictionary

        Returns
        -------
        node_attribute_dict: dict
            The dict of node attributes
        """
        return self._node_attributes

    @property
    def node_features(self) -> NodeFeatView:
        """
        Access node attribute dictionary

        Returns
        -------
        node_attribute_dict: NodeAttrView
            The view of node attributes
        """
        return self.nodes[:].features

    def get_node_attrs(self, nodes: int or slice):
        """
        Get the attributes of the given `nodes`.

        Parameters
        ----------
        nodes: int or slice
            The given node index

        Returns
        -------
        dict
            The node attribute dictionary.
        """
        if isinstance(nodes, slice):
            node_idx = slice_to_list(nodes, self.get_node_num())
        else:
            node_idx = [nodes]

        ret = {}
        for idx in node_idx:
            ret[idx] = self._node_attributes[idx]
        return ret

    def add_edge(self, src: int, tgt: int):
        """
        Add one edge.

        Parameters
        ----------
        src: int
            Source node index
        tgt: int
            Tatget node index

        Returns
        -------
        None
        """
        # Consistency check
        if (src not in range(self.get_node_num())) or (tgt not in range(self.get_node_num())):
            raise NodeNotFoundException('Endpoint not in the graph.')

        # Add edge
        self._edge_indices.src.append(src)
        self._edge_indices.tgt.append(tgt)

    def add_edges(self, src: list, tgt: list):
        """
        Add a bunch of edges to the graph.

        Parameters
        ----------
        src: list of int
            Source node indices
        tgt: list of int
            Target node indices

        Raises
        ------
        SizeMismatchException
            If the lengths of `src` and `tgt` don't match or one of the list contains no element.
        """
        if len(src) == 0:
            raise SizeMismatchException('No endpoint in `src`.')
        elif len(tgt) == 0:
            raise SizeMismatchException('No endpoint in `tgt`.')
        else:
            if len(src) != len(tgt) and len(src) > 1 and len(tgt) > 1:
                raise SizeMismatchException('The numbers of nodes in `src` and `tgt` don\'t match.')
            if len(src) == 1:
                src = [src[0]] * len(tgt)
            elif len(tgt) == 1:
                tgt = [tgt[0]] * len(src)
            for src_idx, tgt_idx in zip(src, tgt):
                self.add_edge(src_idx, tgt_idx)

    def get_edge_num(self) -> int:
        """
        Get the number of edges in the graph

        Returns
        -------
        num_edges: int
            The number of edges
        """
        return len(self._edge_indices.src)

    @property
    def edges(self):
        """
        Return an edge view of the edges and the corresponding data

        Returns
        -------
        edges: EdgeView
        """
        return EdgeView(self)

    def get_all_edges(self):
        """
        Get all the edges in the graph

        Returns
        -------
        edges: list
            List of edges
        """
        edges = list()
        for i in range(self.get_edge_num()):
            edges.append((self._edge_indices.src[i], self._edge_indices.tgt[i]))
        return edges

    def to_dgl(self) -> dgl.DGLGraph:
        """
        Convert to dgl.DGLGraph

        Returns
        -------
        g: dgl.DGLGraph
            The converted dgl.DGLGraph
        """
        dgl_g = dgl.DGLGraph()
        dgl_g.add_nodes(num=self.get_node_num())
        for key, value in self._node_features.items():
            if value is not None:
                dgl_g.ndata[key] = value
        dgl_g.add_edges(u=self._edge_indices.src, v=self._edge_indices.tgt)
        return dgl_g
