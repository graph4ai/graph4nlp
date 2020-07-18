from collections import namedtuple

import dgl
import torch

from graph4nlp.pytorch.data.utils import entail_zero_padding
from graph4nlp.pytorch.data.utils import slice_to_list, SizeMismatchException, NodeNotFoundException
from graph4nlp.pytorch.data.views import NodeView, NodeFeatView, EdgeView

EdgeIndex = namedtuple('EdgeIndex', ['src', 'tgt'])


class GraphData(object):
    """
    Represent a single graph with additional attributes.

    """

    def __init__(self):
        self._node_attributes = dict()
        self._node_features = dict()
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
            self._node_attributes[new_node_idx] = dict()

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
            if key not in self._node_features:  # A new feature is added
                # If the shape of the new feature does not match the number of existing nodes, then error occurs
                if (not isinstance(nodes, slice)) or (
                        len(slice_to_list(nodes, self.get_node_num())) != self.get_node_num()):
                    raise SizeMismatchException(
                        'The new feature should cover all existing {} nodes!'.format(self.get_node_num()))

        # Modification
        for key, value in new_data.items():
            if key not in self._node_features:
                self._node_features[key] = value
            else:
                self._node_features[key][nodes] = value

    # def get_node_attrs(self, nodes):
    #     """
    #     Get node attributes at given nodes.
    #
    #     Parameters
    #     ----------
    #     nodes: int or iterable
    #         The given nodes
    #
    #     Returns
    #     -------
    #     dict
    #
    #     """

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
            raise NodeNotFoundException('End node not in the graph.')

        # Add edge
        self._edge_indices.src.append(src)
        self._edge_indices.tgt.append(tgt)

    def add_edges(self, src: list, tgt: list):
        """
        Add a bunch of edges

        Parameters
        ----------
        src: list of int
            Source node indices

        tgt: list of int
            Target node indices

        Returns
        -------
        None
        """
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
        dgl_g.add_nodes(num=self.get_node_num(), data=self._node_features)
        dgl_g.add_edges(u=self._edge_indices.src, v=self._edge_indices.tgt)
        return dgl_g
