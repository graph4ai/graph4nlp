from collections import namedtuple

import dgl
import numpy as np
import scipy.sparse
import torch

from .utils import SizeMismatchException, NodeNotFoundException, EdgeNotFoundException
from .utils import entail_zero_padding, slice_to_list
from .views import NodeView, NodeFeatView, EdgeView

EdgeIndex = namedtuple('EdgeIndex', ['src', 'tgt'])

node_feat_factory = dict
node_attr_factory = dict
single_node_attr_factory = dict
res_init_node_attr = {'node_attr': None}
res_init_node_features = {'node_feat': None, 'node_emb': None}

eid_nids_mapping_factory = dict
nids_eid_mapping_factory = dict
edge_feature_factory = dict
edge_attribute_factory = dict
single_edge_attr_factory = dict
res_init_edge_features = {'edge_feat': None, 'edge_emb': None}
res_init_edge_attributes = {'edge_attr': None}

graph_data_factory = dict


class GraphData(object):
    """
    Represent a single graph with additional attributes.
    """

    def __init__(self):
        self._node_attributes = node_attr_factory()
        self._node_features = node_feat_factory(res_init_node_features)
        self._edge_indices = EdgeIndex(src=[], tgt=[])
        self._eid_nids_mapping = eid_nids_mapping_factory()
        self._nids_eid_mapping = nids_eid_mapping_factory()
        self._edge_features = edge_feature_factory(res_init_edge_features)
        self._edge_attributes = edge_attribute_factory()
        self.graph_attributes = graph_data_factory()

    # # Graph level data
    # @property
    # def graph_attributes(self):
    #     return self._graph_attributes

    # Node operations
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
            self._node_features[key] = entail_zero_padding(self._node_features[key], node_num)

    # Node feature operations
    @property
    def node_features(self) -> NodeFeatView:
        """
        Access and modify node feature vectors (tensor).
        This property can be accessed in a dict-of-dict fashion, with the order being [name][index].
        'name' indicates the name of the feature vector. 'index' selects the specific nodes to be accessed.
        When accessed independently, returns the feature dictionary with the format {name: tensor}

        Examples
        --------
        >>> g = GraphData()
        >>> g.add_nodes(10)
        >>> import torch
        >>> g.node_features['x'] = torch.rand((10, 10))
        >>> g.node_features['x'][0]
        torch.Tensor([0.1036, 0.6757, 0.4702, 0.8938, 0.6337, 0.3290, 0.6739, 0.1091, 0.7996, 0.0586])


        Returns
        -------
        NodeFeatView
        """

        return self.nodes[:].features

    def get_node_features(self, nodes: int or slice) -> torch.Tensor:
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

    def get_node_feature_names(self):
        return self._node_features.keys()

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
        SizeMismatchException
            If the size of the new features does not match the node number
        """
        # Consistency check
        for key in new_data.keys():
            if key not in self._node_features or self._node_features[key] is None:  # A new feature is added
                # If the shape of the new feature does not match the number of existing nodes, then error occurs
                if (not isinstance(nodes, slice)) or (
                        len(slice_to_list(nodes, self.get_node_num())) != self.get_node_num()):
                    raise SizeMismatchException(
                        'The new feature `{}\' should cover all existing {} nodes!'.format(key, self.get_node_num()))

        # Modification
        for key, value in new_data.items():
            # Node-shape check
            assert value.shape[0] == self.get_node_num(), \
                "The shape feature '{}' does not match the number of nodes in the graph. Got a {} tensor but have {} nodes.".format(
                    key, value.shape, self.get_node_num())

            assert isinstance(value, torch.Tensor), "`{}' is not a tensor. Node features are expected to be tensor."
            if key not in self._node_features or self._node_features[key] is None:
                # self._node_features[key] = None
                self._node_features[key] = value
            else:
                # self._node_features[key][nodes] = None
                if nodes == slice(None, None, None):
                    self._node_features[key] = value
                else:
                    self._node_features[key][nodes] = value

    # Node attribute operations
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

    # Edge views and operations
    @property
    def edges(self):
        """
        Return an edge view of the edges and the corresponding data

        Returns
        -------
        edges: EdgeView
        """
        return EdgeView(self)

    def get_edge_num(self) -> int:
        """
        Get the number of edges in the graph

        Returns
        -------
        num_edges: int
            The number of edges
        """
        return len(self._edge_indices.src)

    def add_edge(self, src: int, tgt: int):
        """
        Add one edge.

        Parameters
        ----------
        src: int
            Source node index
        tgt: int
            Target node index
        """
        # Consistency check
        if (src not in range(self.get_node_num())) or (tgt not in range(self.get_node_num())):
            raise NodeNotFoundException('Endpoint not in the graph.')

        # Append to the mapping list
        endpoint_tuple = (src, tgt)
        eid = self.get_edge_num()
        self._eid_nids_mapping[eid] = endpoint_tuple
        self._nids_eid_mapping[endpoint_tuple] = eid

        # Add edge
        self._edge_indices.src.append(src)
        self._edge_indices.tgt.append(tgt)

        # Initialize edge feature and attribute
        # 1. create placeholder in edge attribute dictionary
        self._edge_attributes[eid] = single_edge_attr_factory(res_init_edge_attributes)
        # 2. perform zero padding
        for key in self._edge_features.keys():
            self._edge_features[key] = entail_zero_padding(self._edge_features[key], 1)

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

    def edge_ids(self, src: int or list, tgt: int or list) -> list:
        """
        Convert the given endpoints to edge indices.

        Parameters
        ----------
        src: int or list
            The index of source node(s).
        tgt: int or list
            The index of target node(s).

        Returns
        -------
        list
            The index of corresponding edges.
        """
        if isinstance(src, int):
            if isinstance(tgt, int):
                try:
                    return [self._nids_eid_mapping[(src, tgt)]]
                except KeyError:
                    raise EdgeNotFoundException('Edge {} does not exist!'.format((src, tgt)))
            elif isinstance(tgt, list):
                eid_list = []
                try:
                    for tgt_idx in tgt:
                        eid_list.append(self._nids_eid_mapping[(src, tgt_idx)])
                except KeyError:
                    raise EdgeNotFoundException('Edge {} does not exist!'.format((src, tgt)))
                return eid_list
            else:
                raise AssertionError("`tgt' must be int or list!")
        elif isinstance(src, list):
            if isinstance(tgt, int):
                eid_list = []
                try:
                    for src_idx in src:
                        eid_list.append(self._nids_eid_mapping[(src_idx, tgt)])
                except KeyError:
                    raise EdgeNotFoundException('Edge {} does not exist!'.format((src, tgt)))
                return eid_list
            elif isinstance(tgt, list):
                if not len(src) == len(tgt):
                    raise SizeMismatchException("The length of `src' and `tgt' don't match!")
                eid_list = []
                try:
                    for src_idx, tgt_idx in zip(src, tgt):
                        eid_list.append(self._nids_eid_mapping[(src_idx, tgt_idx)])
                except KeyError:
                    raise EdgeNotFoundException('Edge {} does not exist!'.format((src, tgt)))
                return eid_list
            else:
                raise AssertionError("`tgt' must be int or list!")
        else:
            raise AssertionError("`src' must be int or list!")

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

    # Edge feature operations
    @property
    def edge_features(self):
        return self.edges[:].features

    def get_edge_feature(self, edges: list):
        """
        Get the feature of the given edges.

        Parameters
        ----------
        edges: list
            Edge indices

        Returns
        -------
        dict
            The dictionary containing all relevant features.
        """
        ret = {}
        for key in self._edge_features.keys():
            if self._edge_features[key] is None:
                ret[key] = None
            else:
                ret[key] = self._edge_features[key][edges]
        return ret

    def get_edge_feature_names(self):
        return self._edge_features.keys()

    def set_edge_feature(self, edges: int or slice or list, new_data: dict):
        """
        Set edge feature

        Parameters
        ----------
        edges: list
            Edge indices
        new_data: dict
            New data

        Raises
        ----------
        SizeMismatchException
            If the size of the new features does not match the node number
        """
        # Consistency check
        for key in new_data.keys():
            if key not in self._edge_features or self._edge_features[key] is None:  # A new feature is added
                # If the shape of the new feature does not match the number of existing nodes, then error occurs
                if (not isinstance(edges, slice)) or (
                        len(slice_to_list(edges, self.get_edge_num())) != self.get_edge_num()):
                    raise SizeMismatchException(
                        'The new feature `{}\' should cover all existing {} edges!'.format(key, self.get_edge_num()))

        # Modification
        for key, value in new_data.items():
            assert isinstance(value, torch.Tensor), "`{}' is not a tensor. Node features are expected to be tensor."
            assert value.shape[0] == self.get_edge_num(), "Length of the feature vector does not match the edge number." \
                                                          "Got tensor '{}' of shape {} but the graph has only {} edges.".format(
                key, value.shape, self.get_edge_num())
            if key not in self._edge_features or self._edge_features[key] is None:
                self._edge_features[key] = value
            else:
                self._edge_features[key][edges] = value

    # Edge attribute operations
    @property
    def edge_attributes(self):
        return self._edge_attributes

    # Conversion utility functions
    def to_dgl(self) -> dgl.DGLGraph:
        """
        Convert to dgl.DGLGraph

        Returns
        -------
        g: dgl.DGLGraph
            The converted dgl.DGLGraph
        """
        dgl_g = dgl.DGLGraph()
        # Add nodes and their features
        dgl_g.add_nodes(num=self.get_node_num())
        for key, value in self._node_features.items():
            if value is not None:
                dgl_g.ndata[key] = value
        # Add edges and their features
        dgl_g.add_edges(u=self._edge_indices.src, v=self._edge_indices.tgt)
        for key, value in self._edge_features.items():
            if value is not None:
                dgl_g.edata[key] = value

        return dgl_g

    def from_dgl(self, dgl_g: dgl.DGLGraph):
        """
        Build the graph from dgl.DGLGraph

        Parameters
        ----------
        dgl_g: dgl.DGLGraph
            The source graph
        """
        assert self.get_edge_num() == 0 and self.get_node_num() == 0, 'Not an empty graph.'

        # Add nodes
        self.add_nodes(dgl_g.number_of_nodes())
        for k, v in dgl_g.ndata.items():
            self.node_features[k] = v

        # Add edges
        src_tensor, tgt_tensor = dgl_g.edges()
        src_list = list(src_tensor.detach().numpy())
        tgt_list = list(tgt_tensor.detach().numpy())
        self.add_edges(src_list, tgt_list)
        for k, v in dgl_g.edata.items():
            self.edge_features[k] = v

    def from_dense_adj(self, adj: torch.Tensor):
        assert adj.dim() == 2, 'Adjancency matrix is not 2-dimensional.'
        assert adj.shape[0] == adj.shape[1], 'Adjancecy is not a square.'

        node_num = adj.shape[0]
        self.add_nodes(node_num)
        edge_weight = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i][j] != 0:
                    self.add_edge(i, j)
                    edge_weight.append(adj[i][j])
        edge_weight = torch.stack(edge_weight, dim=0)
        self.edge_features['edge_weight'] = edge_weight
        # self.edge_features['edge_weight'][-1] = adj[i][j]

    def from_scipy_sparse_matrix(self, adj: scipy.sparse.coo_matrix):
        assert adj.shape[0] == adj.shape[1], 'Adjancecy is not a square.'

        num_nodes = adj.shape[0]
        self.add_nodes(num_nodes)

        for i in range(adj.row.shape[0]):
            self.add_edge(adj.row[i], adj.col[i])
        self.edge_features['edge_weight'] = torch.tensor(adj.data)

    def adj_matrix(self):
        ret = torch.zeros((self.get_node_num(), self.get_node_num()))
        all_edges = self.edges()
        for i in range(len(all_edges)):
            u, v = all_edges[i]
            ret[u][v] = 1
        return ret

    def scipy_sparse_adj(self):
        row = np.array(self._edge_indices[0])
        col = np.array(self._edge_indices[1])
        data = np.ones(self.get_edge_num())
        matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=(self.get_node_num(), self.get_node_num()))
        return matrix

    def union(self, graph):
        """
        Merge a graph into current graph.

        Parameters
        ----------
        graph: GraphData
            The new graph to be merged.
        """

        # Consistency check
        # 0. check if both graphs have nodes
        assert self.get_node_num() > 0 and graph.get_node_num() > 0, \
            "Both participants of the union operation should contain at least 1 node."

        # 1. check if node feature names are consistent
        for feat_name in graph.node_features.keys():
            assert feat_name in self._node_features.keys(), "Node feature '{}' does not exist in current graph.".format(
                feat_name)
            # assert self._node_features[feat_name] is not None, "Node feature '{}' of current graph is None.".format(
            #     feat_name)

        # 2. check if edge feature names are consistent
        for feat_name in graph.edge_features.keys():
            assert feat_name in self._edge_features.keys(), "Edge feature '{}' does not exist in current graph.".format(
                feat_name)
            # assert self._edge_features[feat_name] is not None, "Edge feature '{}' of current graph is None.".format(
            #     feat_name)

        # Save original information
        old_node_num = self.get_node_num()
        old_edge_num = self.get_edge_num()

        # Append information to current graph
        # 1. add nodes
        self.add_nodes(graph.get_node_num())

        # 2. add node features
        append_node_st_idx = old_node_num
        append_node_ed_idx = self.get_node_num()
        for feat_name in graph.node_features.keys():
            if self.node_features[feat_name] is None:
                continue
            self.node_features[feat_name][append_node_st_idx:append_node_ed_idx] = graph.node_features[feat_name]

        # 3. add node attributes
        for node_idx in range(append_node_st_idx, append_node_ed_idx):
            self.node_attributes[node_idx] = graph.node_attributes[node_idx - append_node_st_idx]

        # 4. add edges
        for (src, tgt) in graph.edges():
            self.add_edge(src + old_node_num, tgt + old_node_num)

        # 5. add edge features
        append_edge_st_idx = old_edge_num
        append_edge_ed_idx = self.get_edge_num()
        for feat_name in graph.edge_features.keys():
            if self.edge_features[feat_name] is None:
                continue
            self.edge_features[feat_name][append_edge_st_idx:append_edge_ed_idx] = graph.edge_features[feat_name]

        # 6. add edge attributes
        for edge_idx in range(append_edge_st_idx, append_edge_ed_idx):
            self.edge_attributes[edge_idx] = graph.edge_attributes[edge_idx - append_edge_st_idx]


def to_batch(graphs: list = None) -> GraphData:
    """
    Convert a list of GraphData to a large graph (a batch).

    Parameters
    ----------
    graphs: list of GraphData
        The list of GraphData to be batched

    Returns
    -------
    GraphData
        The large graph containing all the graphs in the batch.
    """
    import copy
    batch = copy.deepcopy(graphs[0])
    batch.batch = [0] * graphs[0].get_node_num()
    for i in range(1, len(graphs)):
        batch.union(graphs[i])
        batch.batch += [i] * graphs[i].get_node_num()
    return batch


def from_batch(batch: GraphData) -> list:
    def rindex(mylist, myvalue):
        # todo: value not found error
        if myvalue not in mylist:
            raise ValueError
        return len(mylist) - mylist[::-1].index(myvalue) - 1

    graphs = []
    batch_size = max(batch.batch) + 1
    # TODO: Consistency check: a graph should contain at least 2 nodes and 1 edges
    # 1. calculate the number of nodes in the batch and get a list indicating #nodes of each graph.
    num_nodes = []
    node_indices = []
    for i in range(batch_size):
        try:
            end_node_index = rindex(batch.batch, i)
        except ValueError:
            ValueError("Graph #{} has no nodes. All graphs in a batch should contain at least one node.".format(i))
        node_indices.append(end_node_index)
        if i == 0:  # the first graph
            num_nodes.append(end_node_index + 1)
        else:
            num_nodes.append(end_node_index - num_nodes[-1])

    # 2. iterate each sub-graph to extract them
    for i in range(batch_size):
        #   a. calculate the starting and ending node index in the batch
        g = GraphData()
        node_st_idx = 0 if i == 0 else node_indices[i - 1]
        node_ed_idx = node_indices[i]
        #   b. extract the corresponding edge indices

        #   c. copy data
        #       i. add nodes
        g.add_nodes(node_ed_idx - node_st_idx)
        #       ii. add node features
        for feat_name in batch.node_features.keys():
            if batch.node_features[feat_name] is None:
                continue
            else:
                g.node_features[feat_name] = batch.node_features[feat_name][node_st_idx:node_ed_idx]
        #       iii. add node attributes
        for i in range(node_st_idx, node_ed_idx):
            g.node_attributes[i - node_st_idx] = batch.node_attributes[i]
        #       iv. add edges
        batch_edges = batch.edges()
        edge_idx = []
        for i in range(len(batch_edges)):
            edge = batch_edges[i]
            if edge[0] in range(node_st_idx, node_ed_idx) and edge[1] in range(node_st_idx, node_ed_idx):
                edge_idx.append(i)
                g.add_edge(edge[0] - node_st_idx, edge[1] - node_st_idx)
        if len(edge_idx) == 0:
            graphs.append(g)
            continue
        edge_st_idx = min(edge_idx)
        edge_ed_idx = max(edge_idx) + 1
        # 5. add edge features
        for feat_name in batch.edge_features.keys():
            if batch.edge_features[feat_name] is None:
                continue
            else:
                g.edge_features[feat_name] = batch.edge_features[feat_name][node_st_idx:node_ed_idx]
        # 6. add edge attributes
        for i in range(edge_st_idx, edge_ed_idx):
            g.edge_attributes[i - edge_st_idx] = batch.edge_attributes[i]
        graphs.append(g)

    assert len(graphs) == batch_size
    return graphs
