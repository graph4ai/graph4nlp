"""
The Graph4NLP library uses the class :py:class:`GraphData` as the representation for structured data (graphs).
:py:class:`GraphData` supports basic operations to the graph, including adding nodes and edges. :py:class:`GraphData` also
supports adding features which are in tensor form, and attributes which are of arbitrary form to the corresponding
nodes or edges. Batching operations is also supported by :py:class:`GraphData`.

"""
import warnings
from collections import namedtuple

import dgl
import numpy as np
import scipy.sparse
import torch

from .utils import SizeMismatchException, EdgeNotFoundException
from .utils import check_and_expand, int_to_list, entail_zero_padding, slice_to_list, reverse_index
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
res_init_edge_features = {'edge_feat': None, 'edge_emb': None, 'edge_weight': None}
res_init_edge_attributes = {'edge_attr': None}

graph_data_factory = dict


class GraphData(object):
    """
    Represent a single graph with additional attributes.
    """

    def __init__(self, src=None, device=None):
        """
        Parameters
        ----------
        src: GraphData, default=None
            The source graph. If not None, then the newly generated graph is a copy of :py:class:`src`.
        device: str, default=None
            The device descriptor for graph. By default it is None.
        """

        # Initialize internal data storages.
        self._node_attributes = node_attr_factory()
        self._node_features = node_feat_factory(res_init_node_features)
        self._edge_indices = EdgeIndex(src=[], tgt=[])
        self._eid_nids_mapping = eid_nids_mapping_factory()
        self._nids_eid_mapping = nids_eid_mapping_factory()
        self._edge_features = edge_feature_factory(res_init_edge_features)
        self._edge_attributes = edge_attribute_factory()
        self.graph_attributes = graph_data_factory()
        self.device = device

        # Batch information. If this instance is not a batch, then the following attributes are all `None`.
        self.batch = None
        self.batch_size = None
        self._batch_num_nodes = None
        self._batch_num_edges = None

        if src is not None:
            if isinstance(src, GraphData):
                self.from_graphdata(src)
            else:
                raise NotImplementedError

    def to(self, device):
        self.device = device
        return self

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
        assert node_num > 0, "The number of nodes to be added should be greater than 0. (Got {})".format(node_num)
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
        Set the features of the `nodes` with the given `new_data``.

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
                    raise ValueError(
                        'The new feature `{}\' should cover all existing {} nodes!'.format(key, self.get_node_num()))

        # Modification
        for key, value in new_data.items():
            # Node-shape check
            assert value.shape[0] == self.get_node_num(), \
                "The shape feature '{}' does not match the number of nodes in the graph. Got a {} tensor but have {} nodes.".format(
                    key, value.shape, self.get_node_num())

            assert isinstance(value, torch.Tensor), "`{}' is not a tensor. Node features are expected to be tensor."
            if key not in self._node_features or self._node_features[key] is None:
                self._node_features[key] = value
            else:
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
        nodes: int
         or slice
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
        Add one edge to the graph.

        Parameters
        ----------
        src: int
            Source node index
        tgt: int
            Target node index

        Raises
        ------
        ValueError
            If one of the endpoints of the edge doesn't exist in the graph.
        """
        # Consistency check
        if (src < 0 or src >= self.get_node_num()) and (tgt < 0 and tgt >= self.get_node_num()):
            raise ValueError('Endpoint not in the graph.')

        # Duplicate edge check. If the edge to be added already exists in the graph, then skip it.
        endpoint_tuple = (src, tgt)
        if endpoint_tuple in self._nids_eid_mapping.keys():
            warnings.warn('Edge {} is already in the graph. Skipping this edge.'.format(endpoint_tuple), Warning)
            return

        # Append to the mapping list
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

    def add_edges(self, src: int or list, tgt: int or list):
        """
        Add a bunch of edges to the graph.

        Parameters
        ----------
        src: int or list
            Source node indices
        tgt: int or list
            Target node indices

        Raises
        ------
        ValueError
            If the lengths of `src` and `tgt` don't match or one of the list contains no element.
        """
        src, tgt = check_and_expand(int_to_list(src), int_to_list(tgt))
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

        Raises
        ------
        EdgeNotFoundException
            If the edge is not in the graph.
        """
        assert isinstance(src, int) or isinstance(src, list), "`src` should be either int or list."
        assert isinstance(tgt, int) or isinstance(tgt, list), "`tgt` should be either int or list."
        src, tgt = check_and_expand(int_to_list(src), int_to_list(tgt))
        eid_list = []
        try:
            for src_idx, tgt_idx in zip(src, tgt):
                eid_list.append(self._nids_eid_mapping[(src_idx, tgt_idx)])
        except KeyError:
            raise EdgeNotFoundException('Edge {} does not exist!'.format((src, tgt)))
        return eid_list

    def get_all_edges(self) -> list:
        """
        Get all the edges in the graph

        Returns
        -------
        edges: list
            List of edges. Each edge is in the shape of the endpoint tuple (src, dst).
        """
        edges = []
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
        """Get all the names of edge features"""
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
            elif edges == slice(None, None, None):
                # Same as node features, if the edges to be modified is all the edges in the graph.
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
        dgl_g = dgl.DGLGraph().to(self.device)
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
        assert self.get_edge_num() == 0 and self.get_node_num() == 0, \
            'This graph isn\'t an empty graph. Please use an empty graph for conversion.'

        # Add nodes
        self.add_nodes(dgl_g.number_of_nodes())
        for k, v in dgl_g.ndata.items():
            self.node_features[k] = v

        # Add edges
        src_tensor, tgt_tensor = dgl_g.edges()
        src_list = list(src_tensor.detach().cpu().numpy())
        tgt_list = list(tgt_tensor.detach().cpu().numpy())
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

    def from_scipy_sparse_matrix(self, adj: scipy.sparse.coo_matrix):
        assert adj.shape[0] == adj.shape[1], 'Got an adjancecy matrix which is not a square.'

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

    def from_graphdata(self, src):
        """Build a clone from a source GraphData"""
        self.add_nodes(src.get_node_num())
        for src_node, tgt_node in src.get_all_edges():
            self.add_edge(src_node, tgt_node)
        for i in range(src.get_node_num()):
            self._node_attributes[i] = src.node_attributes[i]
        for feat_name in src.get_node_feature_names():
            self._node_features[feat_name] = src.node_features[feat_name]
        for i in range(src.get_edge_num()):
            self._edge_attributes[i] = src.edge_attributes[i]
        for feat_name in src.get_edge_feature_names():
            self._edge_features[feat_name] = src.edge_features[feat_name]
        for k, v in src.graph_attributes.items():
            self.graph_attributes[k] = v

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

        # 2. check if edge feature names are consistent
        for feat_name in graph.edge_features.keys():
            assert feat_name in self._edge_features.keys(), "Edge feature '{}' does not exist in current graph.".format(
                feat_name)

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

    def split(self, node_st_idx: int, node_ed_idx: int):
        """
        Given the starting and ending indices of the nodes, split it out of the large graph.

        The corresponding subgraph indicated by the node indices should be a connected component of the large graph and
        should have no connection to other nodes in the graph. Otherwise it cannot be split from the large graph without
        information loss.

        The node indices indicate by `node_st_idx` and `node_ed_idx` is the closed set [`node_st_idx`, `node_ed_idx`].

        Parameters
        ----------
        node_st_idx: int
            The starting node index of the subgraph in the large graph.
        node_ed_idx: int
            The ending node index of the subgraph in the large graph.

        Returns
        -------
        GraphData
            The extracted subgraph.

        Raises
        ------
        ValueError
            If the subgraph has connection with other nodes in the large graph.
        """
        assert node_ed_idx >= node_st_idx, "Got node_ed_idx({}) > node_st_idx({}). " \
                                           "The subgraph should contain at least 1 node.".format(node_ed_idx,
                                                                                                 node_st_idx)
        # extract the corresponding edges from the large graph
        all_edges = self.get_all_edges()
        node_idx_range = range(node_st_idx, node_ed_idx + 1)
        subgraph_edges = []
        for i in range(len(all_edges)):
            current_edge: (int, int) = all_edges[i]
            src, tgt = current_edge
            if src not in node_idx_range and tgt not in node_idx_range:
                continue
            elif src in node_idx_range and tgt in node_idx_range:
                subgraph_edges.append(current_edge)
            else:
                raise ValueError("The subgraph to be extracted has connection with other nodes in the large graph.")
        # convert the edge from node index tuples to edge indices
        no_edge = False
        if len(subgraph_edges) != 0:
            subgraph_edge_src = [src for (src, tgt) in subgraph_edges]
            subgraph_edge_tgt = [tgt for (src, tgt) in subgraph_edges]
            subgraph_edge_ids = self.edge_ids(subgraph_edge_src, subgraph_edge_tgt)
            subgraph_edge_st_idx = min(subgraph_edge_ids)
            subgraph_edge_ed_idx = max(subgraph_edge_ids)
        else:
            no_edge = True
        # build the subgraph
        subgraph = GraphData()
        subgraph.add_nodes(node_ed_idx - node_st_idx + 1)
        for k, v in self._node_features.items():
            if v is not None:
                subgraph.node_features[k] = v[node_st_idx:node_ed_idx + 1]
        for i in range(node_st_idx, node_ed_idx + 1, 1):
            subgraph.node_attributes[i - node_st_idx] = self._node_attributes[i]

        if not no_edge:
            for src, tgt in subgraph_edges:
                subgraph.add_edge(src - node_st_idx, tgt - node_st_idx)
            for k, v in self._edge_features.items():
                if v is not None:
                    subgraph.edge_features[k] = v[subgraph_edge_st_idx:subgraph_edge_ed_idx + 1]
            for i in range(subgraph_edge_st_idx, subgraph_edge_ed_idx + 1, 1):
                subgraph.edge_attributes[i - subgraph_edge_st_idx] = self._edge_attributes[i]
        return subgraph

    def copy_batch_info(self, batch):
        self.batch = batch.batch
        self.device = batch.device
        self.batch_size = batch.batch_size
        self._batch_num_edges = batch._batch_num_edges
        self._batch_num_nodes = batch._batch_num_nodes


def from_dgl(g: dgl.DGLGraph) -> GraphData:
    """
    Convert a dgl.DGLGraph to a GraphData object.

    Parameters
    ----------
    g: dgl.DGLGraph
        The source graph in DGLGraph format.

    Returns
    -------
    GraphData
        The converted graph in GraphData format.
    """
    graph = GraphData()
    graph.from_dgl(g)
    return graph


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
    batch = GraphData(graphs[0], graphs[0].device)
    batch.batch_size = len(graphs)
    batch.batch = [0] * graphs[0].get_node_num()
    batch._batch_num_nodes = [g.get_node_num() for g in graphs]
    batch._batch_num_edges = [g.get_edge_num() for g in graphs]
    for i in range(1, len(graphs)):
        batch.union(graphs[i])
        batch.batch += [i] * graphs[i].get_node_num()
    return batch


def from_batch(batch: GraphData) -> list:
    """
    Convert a batch consisting of several GraphData instances to a list of GraphData instances.

    Parameters
    ----------
    batch: GraphData
        The source batch to be split.

    Returns
    -------
    list
        A list containing all the GraphData instances contained in the source batch.
    """

    num_nodes = batch._batch_num_nodes
    num_edges = batch._batch_num_edges
    all_edges = batch.get_all_edges()
    batch_size = batch.batch_size
    ret = []
    cum_n_nodes = 0
    cum_n_edges = 0

    # Construct graph respectively
    for i in range(batch_size):
        g = GraphData(device=batch.device)
        g.add_nodes(num_nodes[i])
        edges = all_edges[cum_n_edges:cum_n_edges + num_edges[i]]
        src, tgt = [e[0] - cum_n_nodes for e in edges], [e[1] - cum_n_nodes for e in edges]
        g.add_edges(src, tgt)
        cum_n_edges += num_edges[i]
        cum_n_nodes += num_nodes[i]
        ret.append(g)

    # Add node and edge features
    for k, v in batch._node_features.items():
        if v is not None:
            cum_n_nodes = 0  # Cumulative node numbers
            for i in range(batch_size):
                ret[i].node_features[k] = v[cum_n_nodes:cum_n_nodes + num_nodes[i]]
                cum_n_nodes += num_nodes[i]

    for k, v in batch._edge_features.items():
        if v is not None:
            cum_n_edges = 0  # Cumulative edge numbers
            for i in range(batch_size):
                ret[i].edge_features[k] = v[cum_n_edges:cum_n_edges + num_edges[i]]
                cum_n_edges += num_edges[i]
    cum_n_nodes = 0
    cum_n_edges = 0

    # Add node and edge attributes
    for graph_cnt in range(batch_size):
        for num_graph_nodes in range(num_nodes[graph_cnt]):
            ret[graph_cnt].node_attributes[num_graph_nodes] = batch.node_attributes[cum_n_nodes + num_graph_nodes]
        for num_graph_edges in range(num_edges[graph_cnt]):
            ret[graph_cnt].edge_attributes[num_graph_edges] = batch.edge_attributes[cum_n_edges + num_graph_edges]
        cum_n_edges += num_edges[graph_cnt]
        cum_n_nodes += num_nodes[graph_cnt]

    return ret


# Testing code
if __name__ == '__main__':
    a1 = GraphData()
    a1.add_nodes(10)
    a2 = GraphData()
    a2.add_nodes(15)
    a3 = to_batch([a1, a2])
    print(len(a1.node_attributes), len(a2.node_attributes), len(a3.node_attributes))
