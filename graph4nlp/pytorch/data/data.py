"""
The Graph4NLP library uses the class :py:class:`GraphData` as the representation
for structured data (graphs). :py:class:`GraphData` supports basic operations to
the graph, including adding nodes and edges. :py:class:`GraphData` also supports
adding features which are in tensor form, and attributes which are of arbitrary
form to the correspondingnodes or edges. Batching operations is also supported
by :py:class:`GraphData`.
"""
import os
import warnings
from collections import namedtuple
from typing import Any, Callable, Dict, List, Tuple, Union
import dgl
import scipy.sparse
import torch
from torch.nn.utils.rnn import pad_sequence

from .utils import (
    EdgeNotFoundException,
    SizeMismatchException,
    check_and_expand,
    entail_zero_padding,
    int_to_list,
    slice_to_list,
)
from .views import BatchEdgeFeatView, BatchNodeFeatView, EdgeView, NodeFeatView, NodeView

"""
Log level: 0 for verbose, 1 for warnings only, 2 for muted. Default is 0.
"""
log_level = os.environ.get("G4NLP_LOG_LEVEL")
if log_level is None:
    log_level = 0

EdgeIndex = namedtuple("EdgeIndex", ["src", "tgt"])

node_feature_factory = dict
node_attribute_factory = list
single_node_attr_factory = dict
res_init_node_attr = {"node_attr": None}
res_init_node_features = {"node_feat": None, "node_emb": None}

eid_nids_mapping_factory = dict
nids_eid_mapping_factory = dict
edge_feature_factory = dict
edge_attribute_factory = list
single_edge_attr_factory = dict
res_init_edge_features = {"edge_feat": None, "edge_emb": None, "edge_weight": None}
res_init_edge_attributes = {"edge_attr": None}

graph_data_factory = dict


class GraphData(object):
    """
    Represent a heterogeneous graph with additional attributes.
    """

    def __init__(self, src=None, device: str = None, is_hetero: bool = False):
        """
        Parameters
        ----------
        src: GraphData, default=None
            The source graph. If not None, then the newly generated graph
            is a copy of :py:class:`src`.
        device: str, default=None
            The device descriptor for graph. By default it is None.
        is_hetero: bool, default=False
            The heterogeneous graph flag. By default it is False, which means
            the graph is a homogeneous graph (i.e. no node type nor edge type
            specified in the graph). The GraphData instances will include node
            types, edge types, or both if this flag is set to True.
        """

        # Initialize internal data storages.
        self._node_attributes: List[Dict[str, Any]] = node_attribute_factory()
        self._node_features: Dict[str, torch.Tensor] = node_feature_factory(res_init_node_features)
        self._edge_indices = EdgeIndex(src=[], tgt=[])
        self._nids_eid_mapping: Dict[Tuple[int, int], int] = nids_eid_mapping_factory()
        self._edge_features: Dict[str, torch.Tensor] = edge_feature_factory(res_init_edge_features)
        self._edge_attributes: List[Dict[str, Any]] = edge_attribute_factory()
        self.graph_attributes: Dict[str, Any] = graph_data_factory()
        self.device = device
        self.is_hetero = is_hetero

        if is_hetero:
            self._ntypes: List[str] = []
            # self._ntype_to_idx: Dict[str, int] = {}
            # self._idx_to_ntype: Dict[int, str] = {}
            self._etypes: List[Tuple[str, str, str]] = []
        else:
            self._ntypes = None
            self._etypes = None

        # Batch information.
        # If this instance is not a batch, then the following attributes are all `None`.
        self._is_batch = False  # Bool flag indicating whether this graph is a batch graph
        self.batch = None  # Batch node indices
        self.batch_size = None  # Batch size
        self._batch_num_nodes = None  # Subgraph node number list with the length of batch size
        self._batch_num_edges = None  # Subgraph edge number list with the length of batch size

        if src is not None:
            if isinstance(src, GraphData):
                self.from_graphdata(src)
            else:
                raise NotImplementedError

    def to(self, device: str):
        """
        Move the GraphData object to different devices(cpu, gpu, etc.).
        The usage of this method is similar to that of torch.Tensor and
        dgl.DGLGraph

        Parameters
        ----------
        device: str
            The target device.

        Returns
        -------
        self
        """
        self.device = device
        for k, v in self._node_features.items():
            if isinstance(v, torch.Tensor):
                self._node_features[k] = v.to(device)
        for k, v in self._edge_features.items():
            if isinstance(v, torch.Tensor):
                self._edge_features[k] = v.to(device)
        return self

    # Node operations
    @property
    def nodes(self) -> NodeView:
        """
        Return a node view through which the user can access the features
        and attributes.

        A NodeView object provides a high-level view of the underlying
        storage of the features and supports both query and modification
        to the original storage.

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

    def add_nodes(self, node_num: int, ntypes: List[str] = None):
        """
        Add a number of nodes to the graph.

        Parameters
        ------
        node_num: int
            The number of nodes to be added
        """
        if node_num <= 0:
            raise ValueError(
                "The number of nodes to be added should be greater than 0. (Got {})".format(
                    node_num
                )
            )

        if not self.is_hetero:
            if ntypes is not None and len(set(ntypes)) > 1:
                raise ValueError(
                    "The graph is homogeneous, ntypes should be None. Got {}".format(ntypes)
                )
        else:
            if ntypes is None:
                raise ValueError(
                    "The graph is heterogeneous, ntypes should not be None. Got ntypes = {}".format(
                        ntypes
                    )
                )
            if len(ntypes) != node_num:
                raise ValueError(
                    "The number of ntypes should be equal to the number of nodes to be added. "
                    "Got len(ntypes) = {} and node_num = {}".format(len(ntypes), node_num)
                )

        # Create placeholders in the node attribute dictionary
        self._node_attributes.extend(
            [single_node_attr_factory(**res_init_node_attr) for _ in range(node_num)]
        )

        # Do padding in the node feature dictionary
        for key in self._node_features.keys():
            self._node_features[key] = entail_zero_padding(self._node_features[key], node_num)

        # Update the node type information
        if self.is_hetero:
            self._ntypes.extend(ntypes)

    # Node feature operations
    @property
    def node_features(self) -> NodeFeatView:
        """
        Access and modify node feature vectors (tensor).
        This property can be accessed in a dict-of-dict fashion,
        with the order being [name][index].
        'name' indicates the name of the feature vector. 'index' selects
        the specific nodes to be accessed.
        When accessed independently, returns the feature dictionary with
        the format {name: tensor}.

        Examples
        --------
        >>> g = GraphData()
        >>> g.add_nodes(10)
        >>> import torch
        >>> g.node_features['x'] = torch.rand((10, 10))
        >>> g.node_features['x'][0]
        torch.Tensor([0.1036, 0.6757, 0.4702, 0.8938, 0.6337, 0.3290,
                      0.6739, 0.1091, 0.7996, 0.0586])

        Returns
        -------
        NodeFeatView
        """

        return self.nodes[:].features

    @property
    def ntypes(self) -> List[str]:
        """
        Get the node types.

        Returns
        -------
        ntypes: List[str]
            The node types.
        """
        return self._ntypes

    def get_node_features(self, nodes: Union[int, slice]) -> Dict[str, torch.Tensor]:
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

    def node_feature_names(self) -> List[str]:
        """
        Get the names of node features.

        Returns
        -------
        List[str]
            The collection of feature names.
        """
        return list(self._node_features.keys())

    def set_node_features(
        self, nodes: Union[int, slice], new_data: Dict[str, torch.Tensor]
    ) -> None:
        """
        Set the features of the `nodes` with the given `new_data``.

        Parameters
        ----------
        nodes : int or slice
            The nodes involved
        new_data : dict
            The new data to write. Key indicates feature name and value indicates the
            actual value.

        Raises
        ----------
        SizeMismatchException
            If the size of the new features does not match the node number
        """
        # Consistency check
        for key in new_data.keys():
            if (
                key not in self._node_features or self._node_features[key] is None
            ):  # A new feature is added
                # If the shape of the new feature does not match the number of
                # existing nodes, then error occurs
                if (not isinstance(nodes, slice)) or (
                    len(slice_to_list(nodes, self.get_node_num())) != self.get_node_num()
                ):
                    raise ValueError(
                        "The new feature `{}' should cover all existing {} nodes!".format(
                            key, self.get_node_num()
                        )
                    )

        # Modification
        for key, value in new_data.items():
            # Node-shape check
            if not (value.shape[0] == self.get_node_num()):
                raise SizeMismatchException(
                    "The shape feature '{}' does not match the number of nodes in the graph."
                    "Got a {} tensor but have {} nodes.".format(
                        key, value.shape, self.get_node_num()
                    )
                )
            if not isinstance(value, torch.Tensor):
                raise TypeError("torch.Tensor expected, received `{}'.".format(type(value)))

            value_on_device = value.to(self.device)
            if key not in self._node_features or self._node_features[key] is None:
                self._node_features[key] = value_on_device
            else:
                if nodes == slice(None, None, None):
                    self._node_features[key] = value_on_device
                else:
                    self._node_features[key][nodes] = value_on_device

    # Node attribute operations
    @property
    def node_attributes(self) -> List[Any]:
        """
        Access node attribute dictionary

        Returns
        -------
        node_attributes : list
            The list of node attributes
        """
        return self._node_attributes

    def get_node_attrs(self, nodes: Union[int, slice]) -> List[Any]:
        """
        Get the attributes of the given `nodes`.

        Parameters
        ----------
        nodes : int or slice
            The given node index

        Returns
        -------
        list
            The node attribute dictionary.
        """
        return self._node_attributes[nodes]

    # Edge views and operations
    @property
    def edges(self) -> EdgeView:
        """
        Return an edge view of the edges and the corresponding data

        Returns
        -------
        edges : EdgeView
        """
        return EdgeView(self)

    @property
    def etypes(self) -> List[Tuple[str, str, str]]:
        """
        Get the edge types.

        Returns
        -------
        etypes: List[str]
            The edge types.
        """
        return self._etypes

    def get_edge_num(self) -> int:
        """
        Get the number of edges in the graph

        Returns
        -------
        num_edges : int
            The number of edges
        """
        return len(self._edge_indices.src)

    def add_edge(self, src: int, tgt: int, etype: Tuple[str, str, str] = None):
        """
        Add one edge to the graph.

        Parameters
        ----------
        src : int
            Source node index
        tgt : int
            Target node index

        Raises
        ------
        ValueError
            If one of the endpoints of the edge doesn't exist in the graph.
        """
        # Consistency check
        if (src < 0 or src >= self.get_node_num()) and (tgt < 0 and tgt >= self.get_node_num()):
            raise ValueError("Endpoint not in the graph.")
        if self.is_hetero:
            if etype is None:
                raise ValueError("Edge type must be specified for heterograph. Got None.")
        else:
            if etype is not None:
                raise ValueError("Edge type must be None for homograph. Got {}.".format(etype))

        # Duplicate edge check. If the edge to be added already exists in the graph, then skip it.
        endpoint_tuple = (src, tgt)
        if endpoint_tuple in self._nids_eid_mapping.keys():
            if log_level < 2:
                warnings.warn(
                    "Edge {} is already in the graph. Skipping this edge.".format(endpoint_tuple),
                    Warning,
                )
            return

        # Append to the mapping list
        eid = self.get_edge_num()
        self._nids_eid_mapping[endpoint_tuple] = eid

        # Add edge
        self._edge_indices.src.append(src)
        self._edge_indices.tgt.append(tgt)

        # Initialize edge feature and attribute
        # 1. create placeholder in edge attribute dictionary
        self._edge_attributes.append(single_edge_attr_factory(**res_init_edge_attributes))
        # 2. perform zero padding
        for key in self._edge_features.keys():
            self._edge_features[key] = entail_zero_padding(self._edge_features[key], 1)

        # Update edge type
        if etype is not None:
            self._etypes.append(etype)

    def add_edges(
        self,
        src: Union[int, List[int]],
        tgt: Union[int, List[int]],
        etypes: List[Tuple[str, str, str]] = None,
    ) -> None:
        """
        Add a bunch of edges to the graph.

        Parameters
        ----------
        src : int or list
            Source node indices
        tgt : int or list
            Target node indices

        Raises
        ------
        ValueError
            If the lengths of `src` and `tgt` don't match or one of the list contains no element.
        """
        src, tgt = check_and_expand(int_to_list(src), int_to_list(tgt))
        if len(src) != len(tgt):
            raise ValueError(
                "Length of the source and target indices is not the same. "
                "Got {} source nodes and {} target nodes".format(len(src), len(tgt))
            )
        for src_idx, tgt_idx in zip(src, tgt):
            # Consistency check
            if (src_idx < 0 or src_idx >= self.get_node_num()) and (
                tgt_idx < 0 and tgt_idx >= self.get_node_num()
            ):
                raise ValueError("Endpoint not in the graph.")
        # Edge type consistency check
        if self.is_hetero:
            if etypes is None:
                raise ValueError("Edge type must be specified for heterograph. Got None.")
            if len(src) != len(etypes):
                raise ValueError(
                    "Length of edge types and number of edges mismatch."
                    "Got {} edge types and {} edges.".format(len(etypes), len(src))
                )
            for etype in etypes:
                if not isinstance(etype, tuple) and not isinstance(etype, str):
                    raise TypeError(
                        "Edge type must be a tuple of three strings or a single string. \
                            Got {}.".format(
                            type(etype)
                        )
                    )
        else:
            if etypes is not None:
                raise ValueError("Edge type must be None for homograph. Got {}.".format(etypes))

        # Duplicate edge check for non-heterogeneous graph
        if not self.is_hetero:
            duplicate_edge_indices = list()
            for i, endpoint_tuple in enumerate(zip(src, tgt)):
                if endpoint_tuple in self._nids_eid_mapping.keys():
                    if log_level < 2:
                        warnings.warn(
                            f"""Edge {endpoint_tuple} is already in the graph.
                            Since this is a simple graph, we are skipping this edge.""",
                            Warning,
                        )
                    duplicate_edge_indices.append(i)
                    continue
            # Remove duplicate edges
            # Needs to be reversed first to avoid index overflow after popping.
            duplicate_edge_indices.reverse()
            for edge_index in duplicate_edge_indices:
                src.pop(edge_index)
                tgt.pop(edge_index)

        # Append to the mapping list
        current_num_edges = len(self._edge_attributes)
        for i, endpoint_tuple in enumerate(zip(src, tgt)):
            self._nids_eid_mapping[endpoint_tuple] = current_num_edges + i
        num_edges = len(src)

        # Add edge indices
        self._edge_indices.src.extend(src)
        self._edge_indices.tgt.extend(tgt)

        # Initialize edge attributes and features
        self._edge_attributes.extend(
            [single_edge_attr_factory(**res_init_edge_attributes) for _ in range(num_edges)]
        )
        for key in self._edge_features.keys():
            self._edge_features[key] = entail_zero_padding(self._edge_features[key], num_edges)

        # Update etypes
        if etypes is not None:
            if isinstance(etypes[0], str):
                self._etypes.extend(
                    [
                        (self.ntypes[src[i]], etypes[i], self.ntypes[tgt[i]])
                        for i in range(len(etypes))
                    ]
                )
            else:
                self._etypes.extend(etypes)

    def edge_ids(self, src: Union[int, List[int]], tgt: Union[int, List[int]]) -> List[Any]:
        """
        Convert the given endpoints to edge indices.

        Parameters
        ----------
        src : int or list
            The index of source node(s).
        tgt : int or list
            The index of target node(s).

        Returns
        -------
        list
            The index of corresponding edges.

        Raises
        ------
        TypeError
            If the parameters are of wrong types.
        EdgeNotFoundException
            If the edge is not in the graph.
        """
        if not (isinstance(src, int) or isinstance(src, list)):
            raise TypeError("'src' should be either int or list of int.")
        if not (isinstance(tgt, int) or isinstance(tgt, list)):
            raise TypeError("'tgt' should be either int or list of int.")

        src, tgt = check_and_expand(int_to_list(src), int_to_list(tgt))
        eid_list = []
        try:
            for src_idx, tgt_idx in zip(src, tgt):
                if not (isinstance(src_idx, int) or isinstance(tgt_idx, int)):
                    raise TypeError(
                        "'src' and 'tgt' should be (int, int), ('{}', '{}') received.".format(
                            type(src_idx), type(tgt_idx)
                        )
                    )
                eid_list.append(self._nids_eid_mapping[(src_idx, tgt_idx)])
        except KeyError:
            raise EdgeNotFoundException("Edge {} does not exist!".format((src, tgt)))
        return eid_list

    def get_all_edges(self) -> List[Tuple[int, int]]:
        """
        Get all the edges in the graph

        Returns
        -------
        edges : list
            List of edges. Each edge is in the shape of the endpoint tuple (src, dst).
        """
        edges = []
        for i in range(self.get_edge_num()):
            edges.append((self._edge_indices.src[i], self._edge_indices.tgt[i]))
        return edges

    # Edge feature operations
    @property
    def edge_features(self) -> Dict[str, torch.Tensor]:
        """
        Get all the edge features in a dictionary.
        Returns
        -------
        dict
            Edge features with the keys being the feature names and values
            be the corresponding tensors.
        """
        return self.edges[:].features

    def remove_all_edges(self):
        """
        Remove all the edges and the corresponding features and attributes in GraphData.

        Examples
        --------
        >>> g = GraphData()
        >>> g.add_nodes(10)
        >>> g.add_edges(list(range(0, 9, 1)), list(range(1, 10, 1)))
        # Added some feature tensors to the edges
        >>> g.edge_features['random'] = torch.rand((9, 1024, 1024))
        # Remove all edges and the corresponding data. The tensor memory is freed now.
        >>> g.remove_all_edges()

        Returns
        -------
        None
        """

        self._edge_indices = EdgeIndex(src=[], tgt=[])
        self._nids_eid_mapping = nids_eid_mapping_factory()
        self._edge_features = edge_feature_factory(res_init_edge_features)
        self._edge_attributes = edge_attribute_factory()

    def get_edge_feature(self, edges: List[int]) -> Dict[str, torch.Tensor]:
        """
        Get the feature of the given edges.

        Parameters
        ----------
        edges : list
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

    def set_edge_feature(
        self, edges: Union[int, slice, List[int]], new_data: Dict[str, torch.Tensor]
    ):
        """
        Set edge feature

        Parameters
        ----------
        edges : int or list or slice
            Edge indices
        new_data : dict
            New data

        Raises
        ----------
        SizeMismatchException
            If the size of the new features does not match the node number
        """
        # Consistency check
        for key in new_data.keys():
            if key not in self._edge_features or self._edge_features[key] is None:
                # A new feature is added
                # If the shape of the new feature does not match
                # the number of existing nodes, then error occurs
                if (
                    not isinstance(edges, slice)
                    or len(slice_to_list(edges, self.get_edge_num())) != self.get_edge_num()
                ):
                    raise SizeMismatchException(
                        "The new feature `{}' should cover all existing {} edges!".format(
                            key, self.get_edge_num()
                        )
                    )

        # Modification
        for key, value in new_data.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    "'{}' is not a tensor. Node features are expected to be tensor.".format(value)
                )
            if not (value.shape[0] == self.get_edge_num()):
                raise SizeMismatchException(
                    "Length of the feature vector does not match the edge number."
                    "Got tensor '{}' of shape {} but the graph has only {} edges.".format(
                        key, value.shape, self.get_edge_num()
                    )
                )
            # Move the new value to the device consistent with current graph
            value_on_device = value.to(self.device)

            if key not in self._edge_features or self._edge_features[key] is None:
                self._edge_features[key] = value_on_device
            elif edges == slice(None, None, None):
                # Same as node features, if the edges to be modified is all the edges in the graph.
                self._edge_features[key] = value_on_device
            else:
                self._edge_features[key][edges] = value_on_device

    # Edge attribute operations
    @property
    def edge_attributes(self) -> List[Dict[str, torch.Tensor]]:
        """
        Get the edge attributes in a list.
        Returns
        -------
        list:
            A list of dictionaries. Each dictionary represents all
            the attributes on the corresponding edge.
        """
        return self._edge_attributes

    # Conversion utility functions
    @property
    def _data_dict(self) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert the graph data to a dictionary.

        Returns
        -------
        dict
            The dictionary containing all relevant data.
        """
        data_dict = {}
        # 0th pass: Convert typeless node index to typed node index
        typed_node_indices = []
        type_node_counts = {}
        for i in range(self.get_node_num()):
            node_idx_in_type = type_node_counts.setdefault(self._ntypes[i], 0)
            typed_node_indices.append((self._ntypes[i], node_idx_in_type))
            type_node_counts[self._ntypes[i]] += 1
        # 1st pass: scan every edge and make them ((src_type, etype, tgt_type)->(src_idx, tgt_idx))
        for edge_index in range(self.get_edge_num()):
            etype = self.etypes[edge_index]
            key = etype
            src, tgt = self._edge_indices.src[edge_index], self._edge_indices.tgt[edge_index]
            src, tgt = typed_node_indices[src][1], typed_node_indices[tgt][1]
            if key not in data_dict:
                data_dict[key] = ([], [])
            data_dict[key][0].append(src)
            data_dict[key][1].append(tgt)
        # 2nd pass: convert to tensors
        for key, (srcs, tgts) in data_dict.items():
            data_dict[key] = (
                torch.LongTensor(srcs),
                torch.LongTensor(tgts),
            )
        return data_dict

    def to_dgl(self) -> dgl.DGLHeteroGraph:
        """
        Convert to dgl.DGLGraph
        Note that there will be some information loss when calling this function,
        e.g. the batch-related information
        will not be copied to DGLGraph since it is only intended for computation.

        Returns
        -------
        g : dgl.DGLHeteroGraph
            The converted dgl.DGLHeteroGraph
        """
        u, v = self._edge_indices.src, self._edge_indices.tgt
        num_nodes = self.get_node_num()
        # Create DGL graph based on heterogeneity
        if not self.is_hetero:
            dgl_g = dgl.graph(data=(u, v), num_nodes=num_nodes).to(self.device)
            # Add nodes and their features
            for key, value in self._node_features.items():
                if value is not None:
                    dgl_g.ndata[key] = value
            # Add edges and their features
            for key, value in self._edge_features.items():
                if value is not None:
                    dgl_g.edata[key] = value
        else:

            def make_feature_dict(
                features: Dict[str, torch.Tensor],
                item_types: List[Union[str, Tuple[str, str, str]]],
            ) -> Dict[Union[str, Tuple[str, str, str]], Dict[str, torch.Tensor]]:
                """
                To cope with DGL's convention of adding features to heterograph, this function
                converts GraphData's node features to a dictionary of node features indexed by
                type names.

                Parameters
                ----------
                features : torch.Tensor
                    The feature dictionary of GraphData
                item_types : List[str]
                    A list containing the type of each item (node or edge).

                Returns
                -------
                Dict[str, torch.Tensor]
                    A dictionary that can be passed as the edata/ndata argument of DGLGraph.
                """
                ret = {}
                ntype_indices: Dict[str, List[int]] = {}
                # For each type, find the indices of the nodes/edges of that type
                for i, ntype in enumerate(item_types):
                    ntype_index = ntype_indices.setdefault(ntype, [])
                    ntype_index.append(i)
                # Convert indices to tensors
                for n_type, indices in ntype_indices.items():
                    ntype_indices[n_type] = torch.LongTensor(indices)
                # Fill in the features
                for feat_name, feat_value in features.items():
                    ret[feat_name] = {}
                    if feat_value is None:
                        continue
                    for n_type, indices in ntype_indices.items():
                        ret[feat_name][n_type] = feat_value[indices]
                return ret

            def make_num_nodes_dict(
                node_types: List[str],
            ) -> Dict[str, int]:
                """
                To cope with DGL's convention of adding features to heterograph, this function
                converts GraphData's node features to a dictionary of node features indexed by
                type names.

                Parameters
                ----------
                node_types : List[str]
                    A list containing the type of each node.

                Returns
                -------
                Dict[str, int]
                    A dictionary that can be passed as the edata/ndata argument of DGLGraph.
                """
                ret = {}
                for n_type in node_types:
                    if n_type not in ret:
                        ret[n_type] = 0
                    ret[n_type] += 1
                return ret

            data_dict = self._data_dict
            num_nodes_dict = make_num_nodes_dict(self.ntypes)
            dgl_g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict).to(self.device)

            node_hetero_feat_dict = make_feature_dict(self._node_features, self.ntypes)
            edge_hetero_feat_dict = make_feature_dict(self._edge_features, self.etypes)
            num_ntypes = len(set(self.ntypes))
            num_etypes = len(set(self.etypes))
            for feat_name, feat_data_dict in node_hetero_feat_dict.items():
                if len(feat_data_dict) == 0:
                    continue
                if num_ntypes > 1:
                    dgl_g.ndata[feat_name] = feat_data_dict
                else:
                    dgl_g.ndata[feat_name] = feat_data_dict[self.ntypes[0]]
            for feat_name, feat_data_dict in edge_hetero_feat_dict.items():
                if len(feat_data_dict) == 0:
                    continue
                if num_etypes > 1:
                    dgl_g.edata[feat_name] = feat_data_dict
                else:
                    dgl_g.edata[feat_name] = feat_data_dict[self.etypes[0]]

        return dgl_g

    def from_dgl(self, dgl_g: dgl.DGLHeteroGraph, is_hetero=False):
        """
        Build the graph from dgl.DGLHeteroGraph

        Parameters
        ----------
        dgl_g : dgl.DGLHeteroGraph
            The source graph
        """
        if not (self.get_edge_num() == 0 and self.get_node_num() == 0):
            raise ValueError(
                "This graph isn't an empty graph. Please use an empty graph for conversion."
            )
        if not is_hetero:
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
        else:
            self.is_hetero = True
            # For heterogeneous DGL graphs, we perform the same routines for nodes and edges.
            # Specifically, for nodes/edges, we first iterate over all the types.
            # Then we add the nodes and there corresponding features
            # Routine for nodes
            # for ntype in dgl_g.ntypes:
            #     nodes = dgl_g.nodes[ntype]
            #     node_data = nodes.data
            #     if len(node_data) == 0:
            #         warnings.warn("Nodes of type {} have no features. Skipping.".format(ntype))
            #         continue
            #     num_nodes = len(node_data[list(node_data.keys())[0]])
            #     self.add_nodes(num_nodes, ntypes=[ntype] * num_nodes)
            #     for feature_name, feature_value in node_data.items():
            #         self.node_features[feature_name] = feature_value
            node_data = dgl_g.ndata
            ntypes = []
            processed_node_types = False
            node_feat_dict = {}
            for feature_name, data_dict in node_data.items():
                if not isinstance(data_dict, Dict):  
                    # DGL will return tensor if ntype is single
                    # This can happen when graph is a multigraph
                    data_dict = {dgl_g.ntypes[0]: data_dict}
                if not processed_node_types:
                    for node_type, node_feature in data_dict.items():
                        ntypes += [node_type] * len(node_feature)
                    processed_node_types = True
                # for node_type, node_feature in data_dict.items():
                node_feat_dict[feature_name] = torch.cat(list(data_dict.values()), dim=0)
            self.add_nodes(len(ntypes), ntypes=ntypes)
            for feature_name, feature_value in node_feat_dict.items():
                self.node_features[feature_name] = feature_value
            # do the same thing for edges
            dgl_g_etypes = dgl_g.canonical_etypes
            # Add edges first
            edge_feature_dict = {}
            for etype in dgl_g_etypes:
                num_edges = dgl_g.num_edges(etype)
                src_type, r_type, dst_type = etype
                srcs, dsts = dgl_g.find_edges(
                    torch.tensor(list(range(num_edges)), dtype=torch.long, device=dgl_g.device),
                    etype,
                )
                srcs, dsts = (
                    srcs.detach().cpu().numpy().tolist(),
                    dsts.detach().cpu().numpy().tolist(),
                )
                self.add_edges(srcs, dsts, etypes=[etype] * num_edges)
                if len(dgl_g_etypes) > 1:
                    for feature_name, feature_dict in dgl_g.edata.items():
                        current_feat = edge_feature_dict.get(feature_name, None)
                        if current_feat is None:
                            current_feat = feature_dict[etype]
                        else:
                            current_feat = torch.cat([current_feat, feature_dict[etype]], dim=0)
                        edge_feature_dict[feature_name] = current_feat
                else:
                    for feature_name, feature_value in dgl_g.edata.items():
                        edge_feature_dict[feature_name] = feature_value
            # Add edge features then
            for feat_name, feat_value in edge_feature_dict.items():
                self.edge_features[feat_name] = feat_value
            # edge_data = dgl_g.edata
            # etypes = []
            # processed_edge_types = False
            # edge_feat_dict = {}
            # for feature_name, data_dict in edge_data.items():
            #     if not processed_edge_types:
            #         for edge_type, edge_feature in data_dict.items():
            #             etypes += [edge_type] * len(edge_feature)
            #         processed_edge_types = True
            #     # for edge_type, edge_feature in data_dict.items():
            #     edge_feat_dict[feature_name] = torch.stack(list(data_dict.values()), dim=0)
            # self.add_edges()
            # for feature_name, feature_value in edge_feat_dict.items():
            #     self.edge_features[feature_name] = feature_value

        return self

    def from_dense_adj(self, adj: torch.Tensor):
        """
        Construct a graph from a dense (2-D NxN) adjacency matrix with
        the edge weights represented by the value of the matrix entries.

        Parameters
        ----------
        adj : torch.Tensor
            The tensor representing the adjacency matrix.

        Returns
        -------
        self
        """
        if not (adj.dim() == 2):
            raise SizeMismatchException("Adjancency matrix is not 2-dimensional.")
        if not (adj.shape[0] == adj.shape[1]):
            raise SizeMismatchException("Adjancecy is not a square.")

        node_num = adj.shape[0]
        self.add_nodes(node_num)
        edge_weight = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i][j] != 0:
                    self.add_edge(i, j)
                    edge_weight.append(adj[i][j])
        edge_weight = torch.stack(edge_weight, dim=0)
        self.edge_features["edge_weight"] = edge_weight
        return self

    def from_scipy_sparse_matrix(self, adj: scipy.sparse.coo_matrix):
        """
        Construct a graph from a sparse adjacency matrix with the edge weights
        represented by the value of the matrix entries.
        Parameters
        ----------
        adj : scipy.sparse.coo_matrix
            The object representing the sparse adjacency matrix.

        Returns
        -------
        self
        """
        if not (adj.shape[0] == adj.shape[1]):
            raise SizeMismatchException("Got an adjancecy matrix which is not a square.")

        num_nodes = adj.shape[0]
        self.add_nodes(num_nodes)

        for i in range(adj.row.shape[0]):
            self.add_edge(adj.row[i], adj.col[i])
        self.edge_features["edge_weight"] = torch.tensor(adj.data)
        return self

    def adj_matrix(
        self, batch_view: bool = False, post_processing_fn: Callable = None
    ) -> torch.Tensor:
        """
        Returns the adjacency matrix of the graph. Returns a 2D tensor if
        it is a single graph and a 3D tensor if it is a batched graph, with
        the matrices padded with 0 (B x N x N)

        Parameters
        ----------
        batch_view : bool
            Whether to return a batched view of the adjacency matrix(3D, True) or not (2D).
        post_processing_fn : function
            A callback function which takes a binary adjacency matrix (2D)
            and do some post-processing on it.
            The return of this function should also be N x N

        Returns
        -------
        torch.Tensor:
            The adjacency matrix (N x N if batch_view=False and B x N x N if batch_view=True).
        """
        if batch_view is False:
            ret = torch.zeros((self.get_node_num(), self.get_node_num())).to(self.device)
            all_edges = self.edges()
            for i in range(len(all_edges)):
                u, v = all_edges[i]
                ret[u][v] = 1
            if post_processing_fn is not None:
                ret = post_processing_fn(ret)
            return ret
        else:
            if not self._is_batch:
                raise Exception("Cannot enable batch view on a non-batch graph!")
            max_num_nodes = max(self._batch_num_nodes)
            ret = torch.zeros((self.batch_size, max_num_nodes, max_num_nodes)).to(self.device)
            cum_num_nodes = 0
            cum_num_edges = 0
            all_edges = self.get_all_edges()
            for i in range(self.batch_size):
                edges = all_edges[cum_num_edges : cum_num_edges + self._batch_num_edges[i]]
                for edge in edges:
                    ret[i][edge[0] - cum_num_nodes, edge[1] - cum_num_nodes] = 1
                if post_processing_fn is not None:
                    ret[i] = post_processing_fn(ret[i])
                cum_num_nodes += self._batch_num_nodes[i]
                cum_num_edges += self._batch_num_edges[i]
            return ret

    def sparse_adj(self, batch_view: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Return the scipy.sparse.coo_matrix form of the adjacency matrix
        Parameters
        ----------
        batch_view : bool
            Whether to return the split view of the adjacency matrix.
            Return a list of COO matrix if True

        Returns
        -------
        torch.Tensor or list of torch.Tensor
        """
        row = torch.tensor(self._edge_indices[0]).to(self.device)
        col = torch.tensor(self._edge_indices[1]).to(self.device)
        data = torch.ones(self.get_edge_num()).to(self.device)
        if not batch_view:
            indices = torch.stack([row, col])
            matrix = torch.sparse_coo_tensor(
                indices=indices, values=data, size=(self.get_node_num(), self.get_node_num())
            )
            return matrix
        else:
            if self._is_batch is not True:
                raise Exception(
                    "Cannot enable batch view of COO adjacency matrix on a non-batch graph."
                )
            matrices = []
            cum_num_edges = 0
            cum_num_nodes = 0
            for i in range(self.batch_size):
                num_edges = self._batch_num_edges[i]
                num_nodes = self._batch_num_nodes[i]
                # Slicing the matrix one by one
                cur_row = row[cum_num_edges : cum_num_edges + num_edges]
                cur_col = col[cum_num_edges : cum_num_edges + num_edges]
                cur_data = data[cum_num_edges : cum_num_edges + num_edges]
                cur_row -= cum_num_nodes
                cur_col -= cum_num_nodes
                indices = torch.stack([cur_row, cur_col])
                cur_matrix = torch.sparse_coo_tensor(
                    indices=indices, values=cur_data, size=(num_nodes, num_nodes)
                )
                matrices.append(cur_matrix)
                cum_num_edges += num_edges
                cum_num_nodes += num_nodes
            return matrices

    def from_graphdata(self, src: Any):
        """Build a clone from a source GraphData"""

        # Add nodes and edges
        self.add_nodes(src.get_node_num())
        self.add_edges(src._edge_indices.src, src._edge_indices.tgt)

        # Deepcopy of feature tensors
        for k, v in src._node_features.items():
            self._node_features[k] = v
        for k, v in src._edge_features.items():
            self._edge_features[k] = v

        # Copy attributes
        import copy

        self._node_attributes = copy.deepcopy(src.node_attributes)
        self._edge_attributes = copy.deepcopy(src.edge_attributes)
        self.graph_attributes = copy.deepcopy(src.graph_attributes)

        # Copy batch information if necessary
        if src._is_batch:
            self.copy_batch_info(src)

        # Move data to the device of the source graph
        self.to(src.device)

    def copy_batch_info(self, batch: Any) -> None:
        """
        Copy all the information related to the batching.
        Parameters
        ----------
        batch:
            The source batch from which the information comes.

        Returns
        -------
        None
        """
        self._is_batch = True
        self.batch = batch.batch
        self.device = batch.device
        self.batch_size = batch.batch_size
        self._batch_num_edges = batch._batch_num_edges
        self._batch_num_nodes = batch._batch_num_nodes

    @property
    def batch_node_features(self):
        """
        Get a view of the batched(padded) version of the node features. Shape: (B, N, D)

        Returns
        -------
        BatchNodeFeatView
        """
        return BatchNodeFeatView(self)

    def _get_batch_node_features(
        self, item: str = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get the batched view of node feature tensors, i.e., tensors in (B, N, D) view

        Parameters
        -------
        item : str
            The name of the features. If None then return a dictionary of all the features.

        Returns
        -------
        dict or torch.Tensor
            A dictionary containing the node feature names and the corresponding
            batch-view tensors, or just the specified tensor.
        """
        if not self._is_batch:
            raise Exception("Calling batch_node_features() method on a non-batch graph.")
        if item is None:
            batch_node_features = dict()
            separate_features = self.split_node_features
            for k, v in separate_features.items():
                batch_node_features[k] = pad_sequence(list(v), batch_first=True)
            return batch_node_features
        else:
            if (item not in self.node_features.keys()) or (self.node_features[item] is None):
                raise Exception("Node feature {} doesn't exist!".format(item))
            return pad_sequence(self.split_node_features[item], batch_first=True)

    def _set_batch_node_features(self, key: str, value: torch.Tensor):
        """
        Set node features in batch view.

        Parameters
        ----------
        key : str
            The name of the feature.
        value : torch.Tensor
            The values to be written, in the shape of (B, N, D)
        """
        individual_features = [
            value[i, : self._batch_num_nodes[i]] for i in range(len(self._batch_num_nodes))
        ]
        self.set_node_features(slice(None, None, None), {key: torch.cat(individual_features)})

    @property
    def batch_edge_features(self) -> BatchEdgeFeatView:
        """
        Edge version of self.batch_node_features

        Returns
        -------
        BatchEdgeFeatView
        """
        return BatchEdgeFeatView(self)

    def _get_batch_edge_features(
        self, item: str = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        An edge version of :py:method `batch_node_features`.

        Returns
        -------
        dict or torch.Tensor
            A dictionary containing the edge feature names and the corresponding batch-view tensors.
        """
        if not self._is_batch:
            raise Exception("Calling batch_edge_features() method on a non-batch graph.")
        if item is None:
            batch_edge_features = dict()
            for k, v in self.split_edge_features.items():
                batch_edge_features[k] = pad_sequence(list(v), batch_first=True)
            return batch_edge_features
        else:
            if (item not in self.edge_features.keys()) or (self.edge_features[item] is None):
                raise Exception("Edge feature {} doesn't exist!".format(item))
            return pad_sequence(self.split_edge_features[item], batch_first=True)

    def _set_batch_edge_features(self, key: str, value: torch.Tensor):
        individual_features = [
            value[i, : self._batch_num_edges[i]] for i in range(len(self._batch_num_edges))
        ]
        self.set_edge_feature(slice(None, None, None), {key: torch.cat(individual_features)})

    @property
    def split_node_features(self) -> Dict[str, Tuple[torch.Tensor]]:
        if not self._is_batch:
            raise Exception("Calling split_node_features() method on a non-batch graph.")
        node_features = dict()
        for feature in self.node_features.keys():
            if self.node_features[feature] is None:
                continue
            node_features[feature] = torch.split(
                self.node_features[feature], split_size_or_sections=self._batch_num_nodes
            )
        return node_features

    @property
    def split_edge_features(self) -> Dict[str, Tuple[torch.Tensor]]:
        if not self._is_batch:
            raise Exception("Calling split_edge_features() method on a non-batch graph.")
        edge_features = dict()
        for feature in self.edge_features.keys():
            if self.edge_features[feature] is None:
                continue
            edge_features[feature] = torch.split(
                self.edge_features[feature], split_size_or_sections=self._batch_num_edges
            )
        return edge_features

    def split_features(self, input_tensor: torch.Tensor, type: str = "node") -> torch.Tensor:
        """
        Convert a tensor from [N, *] to [B, N_max, *] with zero padding according
        to the batch information stored in the graph.

        Parameters
        ----------
        input_tensor : torch.Tensor
        The original tensor to be split.
        type : str
        'node' or 'edge'. Indicates the source of batch information.

        Returns
        -------
        torch.Tensor
        The split tensor.
        """
        input_tensor = input_tensor.to(self.device)
        if not self._is_batch:
            raise Exception("Cannot invoke `batch_split` method on a non-batch graph.")
        if type == "node":
            info_src = self._batch_num_nodes
        elif type == "edge":
            info_src = self._batch_num_edges
        else:
            raise NotImplementedError(
                "Currently only 'node' and 'edge' is accepted in GraphData.split_features()."
            )
        num_instance = 0
        for number in info_src:
            num_instance += number
        if not (num_instance == input_tensor.shape[0]):
            raise SizeMismatchException(
                "Number of instances "
                "doesn't match: The graph "
                "has {} instances while the input "
                "contains {}".format(num_instance, input_tensor.shape[0])
            )
        n_max = max(info_src)
        output = torch.zeros(size=(self.batch_size, n_max, *input_tensor.shape[1:])).to(self.device)
        split_input = torch.split(tensor=input_tensor, split_size_or_sections=info_src)
        for i in range(self.batch_size):
            output[i, : info_src[i]] = split_input[i]
        return output


def from_dgl(g: dgl.DGLGraph) -> GraphData:
    """
    Convert a dgl.DGLGraph to a GraphData object.

    Parameters
    ----------
    g : dgl.DGLGraph
        The source graph in DGLGraph format.

    Returns
    -------
    GraphData
        The converted graph in GraphData format.
    """
    dgl_g_is_hetero = (not g.is_homogeneous) or g.is_multigraph
    graph = GraphData(is_hetero=dgl_g_is_hetero)
    graph.from_dgl(g, is_hetero=dgl_g_is_hetero)
    return graph


def to_batch(graphs: List[GraphData] = None) -> GraphData:
    """
    Convert a list of GraphData to a large graph (a batch).

    Parameters
    ----------
    graphs : list of GraphData
        The list of GraphData to be batched

    Returns
    -------
    GraphData
        The large graph containing all the graphs in the batch.
    """

    # Check
    if not isinstance(graphs, list):
        raise TypeError("to_batch() only accepts list of GraphData!")
    if not (len(graphs) > 0):
        raise ValueError("Cannot convert an empty list of graphs into a big batched graph!")
    is_heterograph = graphs[0].is_hetero
    if not (all([g.is_hetero for g in graphs]) == is_heterograph):
        raise ValueError("All the graphs in the batch must all be heterogeneous or homogeneous!")

    # Optimized version
    big_graph = GraphData(is_hetero=is_heterograph)
    big_graph._is_batch = True
    big_graph.device = graphs[0].device

    total_num_nodes = 0
    for g in graphs:
        total_num_nodes += g.get_node_num()

    # Step 1: Add nodes and node types
    node_types = None
    if is_heterograph:
        node_types = []
        for g in graphs:
            node_types += g.ntypes
    big_graph.add_nodes(total_num_nodes, ntypes=node_types)

    # Step 2: Set node features
    node_features = dict()
    for g in graphs:
        for feature_name in g.node_features.keys():
            if feature_name in node_features:
                node_features[feature_name].append(g.node_features[feature_name])
            else:
                node_features[feature_name] = [g.node_features[feature_name]]
    for k, v in node_features.items():
        if None in v:
            continue
        else:
            feature_tensor = torch.cat(v, dim=0)
            big_graph.node_features[k] = feature_tensor

    # Step 3: Set node attributes
    total_node_count = 0
    for g in graphs:
        for i in range(g.get_node_num()):
            big_graph.node_attributes[total_node_count] = g.node_attributes[i]
            total_node_count += 1

    # Step 4: Add edges and etypes
    def stack_edge_indices(gs):
        all_edge_indices = EdgeIndex(src=[], tgt=[])
        cumulative_node_num = 0
        for g in gs:
            for edge_index_tuple in g.get_all_edges():
                src, tgt = edge_index_tuple
                src += cumulative_node_num
                tgt += cumulative_node_num
                all_edge_indices.src.append(src)
                all_edge_indices.tgt.append(tgt)
            cumulative_node_num += g.get_node_num()
        return all_edge_indices

    all_edge_indices = stack_edge_indices(graphs)
    edge_types = None
    if is_heterograph:
        edge_types = []
        for g in graphs:
            edge_types += g.etypes
    big_graph.add_edges(all_edge_indices.src, all_edge_indices.tgt, etypes=edge_types)

    # Step 5: Add edge features
    edge_features = dict()
    for g in graphs:
        for feature_name in g.edge_features.keys():
            if feature_name in edge_features:
                edge_features[feature_name].append(g.edge_features[feature_name])
            else:
                edge_features[feature_name] = [g.edge_features[feature_name]]
    for k, v in edge_features.items():
        if None in v:
            continue
        else:
            feature_tensor = torch.cat(v, dim=0)
            big_graph.edge_features[k] = feature_tensor

    # Step 6: Add edge attributes
    total_edge_count = 0
    for g in graphs:
        for i in range(g.get_edge_num()):
            big_graph.edge_attributes[total_edge_count] = g.edge_attributes[i]
            total_edge_count += 1

    # Step 7: Batch information preparation
    big_graph.batch_size = len(graphs)
    batch_numbers = []
    for i in range(len(graphs)):
        batch_numbers.extend([i] * graphs[i].get_node_num())
    big_graph.batch = batch_numbers
    big_graph._batch_num_nodes = [g.get_node_num() for g in graphs]
    big_graph._batch_num_edges = [g.get_edge_num() for g in graphs]

    # Step 8: merge node and edge types if the batch is heterograph
    # if is_heterograph:
    #     node_types = []
    #     edge_types = []
    #     for g in graphs:
    #         node_types.extend(g.node_types)
    #         edge_types.extend(g.edge_types)
    #     big_graph.node_types = node_types
    #     big_graph.edge_types = edge_types
    return big_graph


def from_batch(batch: GraphData) -> List[GraphData]:
    """
    Convert a batch consisting of several GraphData instances to a list of GraphData instances.

    Parameters
    ----------
    batch : GraphData
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
    is_hetero = batch.is_hetero
    ret: List[GraphData] = []
    cum_n_nodes = 0
    cum_n_edges = 0

    # Construct graph respectively
    for i in range(batch_size):
        g = GraphData(device=batch.device, is_hetero=is_hetero)
        g_ntypes = batch.ntypes[cum_n_nodes : cum_n_nodes + num_nodes[i]] if is_hetero else None
        g.add_nodes(num_nodes[i], ntypes=g_ntypes)
        edges = all_edges[cum_n_edges : cum_n_edges + num_edges[i]]
        g_etypes = batch.etypes[cum_n_edges : cum_n_edges + num_edges[i]] if is_hetero else None
        src, tgt = [e[0] - cum_n_nodes for e in edges], [e[1] - cum_n_nodes for e in edges]
        g.add_edges(src, tgt, etypes=g_etypes)
        cum_n_edges += num_edges[i]
        cum_n_nodes += num_nodes[i]
        ret.append(g)

    # Add node and edge features
    for k, v in batch._node_features.items():
        if v is not None:
            cum_n_nodes = 0  # Cumulative node numbers
            for i in range(batch_size):
                ret[i].node_features[k] = v[cum_n_nodes : cum_n_nodes + num_nodes[i]]
                cum_n_nodes += num_nodes[i]

    for k, v in batch._edge_features.items():
        if v is not None:
            cum_n_edges = 0  # Cumulative edge numbers
            for i in range(batch_size):
                ret[i].edge_features[k] = v[cum_n_edges : cum_n_edges + num_edges[i]]
                cum_n_edges += num_edges[i]

    # Add node and edge attributes
    cum_n_nodes = 0
    cum_n_edges = 0
    for graph_cnt in range(batch_size):
        for num_graph_nodes in range(num_nodes[graph_cnt]):
            ret[graph_cnt].node_attributes[num_graph_nodes] = batch.node_attributes[
                cum_n_nodes + num_graph_nodes
            ]
        for num_graph_edges in range(num_edges[graph_cnt]):
            ret[graph_cnt].edge_attributes[num_graph_edges] = batch.edge_attributes[
                cum_n_edges + num_graph_edges
            ]
        cum_n_edges += num_edges[graph_cnt]
        cum_n_nodes += num_nodes[graph_cnt]

    # Add node and edge types
    if batch.is_hetero:
        cum_n_nodes = 0
        cum_n_edges = 0
        for graph_cnt in range(batch_size):
            ntypes = batch.ntypes[cum_n_nodes : cum_n_nodes + num_nodes[graph_cnt]]
            ret[graph_cnt]._ntypes = ntypes
            cum_n_nodes += num_nodes[graph_cnt]
            etypes = batch.etypes[cum_n_edges : cum_n_edges + num_edges[graph_cnt]]
            ret[graph_cnt]._etypes = etypes
            cum_n_edges += num_edges[graph_cnt]

    return ret


# Testing code
if __name__ == "__main__":
    a1 = GraphData()
    a1.add_nodes(10)
    a2 = GraphData()
    a2.add_nodes(15)
    a3 = to_batch([a1, a2])
    print(len(a1.node_attributes), len(a2.node_attributes), len(a3.node_attributes))
