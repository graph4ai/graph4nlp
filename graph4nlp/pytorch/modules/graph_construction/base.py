import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import nn

from ..utils.constants import INF, VERY_SMALL_NUMBER
from ..utils.generic_utils import normalize_adj, sparse_mx_to_torch_sparse_tensor


class StaticGraphConstructionBase:
    """
    Base class for static graph construction.
    """

    def __init__(self):
        super(StaticGraphConstructionBase, self).__init__()

    def add_vocab(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    def parsing(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    def static_topology(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _construct_static_graph(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _graph_connect(cls, **kwargs):
        raise NotImplementedError()


class DynamicGraphConstructionBase(nn.Module):
    """Base class for dynamic graph construction.

    Parameters
    ----------
    sim_metric_type : str, optional
        Specify similarity metric function type including "attention",
        "weighted_cosine", "gat_attention", "rbf_kernel", and "cosine".
        Default: ``"weighted_cosine"``.
    num_heads : int, optional
        Specify the number of heads for multi-head similarity metric
        function, default: ``1``.
    top_k_neigh : int, optional
        Specify the top k value for knn neighborhood graph sparsificaiton,
        default: ``None``.
    epsilon_neigh : float, optional
        Specify the epsilon value (i.e., between ``0`` and ``1``) for
        epsilon neighborhood graph sparsificaiton, default: ``None``.
    smoothness_ratio : float, optional
        Specify the smoothness ratio (i.e., between ``0`` and ``1``)
        for graph regularization on smoothness, default: ``None``.
    connectivity_ratio : float, optional
        Specify the connectivity ratio (i.e., between ``0`` and ``1``)
        for graph regularization on connectivity, default: ``None``.
    sparsity_ratio : float, optional
        Specify the sparsity ratio (i.e., between ``0`` and ``1``)
        for graph regularization on sparsity, default: ``None``.
    input_size : int, optional
        The dimension of input embeddings, default: ``None``.
    hidden_size : int, optional
        The dimension of hidden layers, default: ``None``.
    """

    def __init__(
        self,
        sim_metric_type="weighted_cosine",
        num_heads=1,
        top_k_neigh=None,
        epsilon_neigh=None,
        smoothness_ratio=None,
        connectivity_ratio=None,
        sparsity_ratio=None,
        input_size=None,
        hidden_size=None,
    ):
        super(DynamicGraphConstructionBase, self).__init__()
        assert (
            top_k_neigh is None or epsilon_neigh is None
        ), "top_k_neigh and epsilon_neigh cannot be activated at the same time!"
        self.top_k_neigh = top_k_neigh
        self.epsilon_neigh = epsilon_neigh
        self.sim_metric_type = sim_metric_type
        self.smoothness_ratio = smoothness_ratio
        self.connectivity_ratio = connectivity_ratio
        self.sparsity_ratio = sparsity_ratio

        if self.sim_metric_type == "attention":
            self.mask_off_val = -INF
            self.linear_sims = nn.ModuleList(
                [nn.Linear(input_size, hidden_size, bias=False) for _ in range(num_heads)]
            )
        elif self.sim_metric_type == "weighted_cosine":
            self.mask_off_val = 0
            self.weight = torch.Tensor(num_heads, input_size)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        elif self.sim_metric_type == "gat_attention":
            self.mask_off_val = -INF
            self.linear_sims1 = nn.ModuleList(
                [nn.Linear(input_size, 1, bias=False) for _ in range(num_heads)]
            )
            self.linear_sims2 = nn.ModuleList(
                [nn.Linear(input_size, 1, bias=False) for _ in range(num_heads)]
            )
            self.leakyrelu = nn.LeakyReLU(0.2)
        elif self.sim_metric_type == "rbf_kernel":
            self.mask_off_val = 0
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(
                nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size))
            )
        elif self.sim_metric_type == "cosine":
            self.mask_off_val = 0
        else:
            raise RuntimeError("Unknown sim_metric_type: {}".format(self.sim_metric_type))

    def dynamic_topology(self, node_emb, edge_emb=None, init_adj=None, node_mask=None, **kwargs):
        """Compute graph topology.

        Parameters
        ----------
        node_emb : torch.Tensor
            The node embeddings.
        edge_emb : torch.Tensor, optional
            The edge embeddings, default: ``None``.
        init_adj : torch.Tensor, optional
            The initial adjacency matrix, default: ``None``.
        node_mask : torch.Tensor, optional
            The node mask matrix, default: ``None``.
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

    def compute_similarity_metric(self, node_emb, node_mask=None):
        """Compute similarity metric.

        Parameters
        ----------
        node_emb : torch.Tensor
            The input node embedding matrix.
        node_mask : torch.Tensor, optional
            The node mask matrix, default: ``None``.

        Returns
        -------
        torch.Tensor
            Adjacency matrix.
        """
        if self.sim_metric_type == "attention":
            attention = 0
            for _ in range(len(self.linear_sims)):
                node_vec_t = torch.relu(self.linear_sims[_](node_emb))
                attention += torch.matmul(node_vec_t, node_vec_t.transpose(-1, -2))

            attention /= len(self.linear_sims)
        elif self.sim_metric_type == "weighted_cosine":
            expand_weight_tensor = self.weight.unsqueeze(1)
            if len(node_emb.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            node_vec_t = node_emb.unsqueeze(0) * expand_weight_tensor
            node_vec_norm = F.normalize(node_vec_t, p=2, dim=-1)
            attention = torch.matmul(node_vec_norm, node_vec_norm.transpose(-1, -2)).mean(0)
        elif self.sim_metric_type == "gat_attention":
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](node_emb)
                a_input2 = self.linear_sims2[_](node_emb)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
        elif self.sim_metric_type == "rbf_kernel":
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self._compute_distance_matrix(node_emb, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis ** 2))
        elif self.sim_metric_type == "cosine":
            node_vec_norm = node_emb.div(torch.norm(node_emb, p=2, dim=-1, keepdim=True))
            attention = torch.mm(node_vec_norm, node_vec_norm.transpose(-1, -2)).detach()

        if node_mask is not None:
            if torch.__version__ < "1.3.0":
                attention = attention.masked_fill_(~(node_mask == 1.0), self.mask_off_val)
            else:
                attention = attention.masked_fill_(~node_mask.bool(), self.mask_off_val)

        return attention

    def sparsify_graph(self, adj):
        """Return a sparsified graph of the input graph. The graph sparsification strategy
        is determined by ``top_k_neigh`` and ``epsilon_neigh``.

        Parameters
        ----------
        adj : torch.Tensor
            The input adjacency matrix.

        Returns
        -------
        torch.Tensor
            The output adjacency matrix.
        """
        if self.epsilon_neigh is not None:
            adj = self._build_epsilon_neighbourhood(adj, self.epsilon_neigh)

        if self.top_k_neigh is not None:
            adj = self._build_knn_neighbourhood(adj, self.top_k_neigh)

        return adj

    def compute_graph_regularization(self, adj, node_feat):
        """Graph graph regularization loss.

        Parameters
        ----------
        adj : torch.Tensor
            The adjacency matrix.
        node_feat : torch.Tensor
            The node feature matrix.

        Returns
        -------
        torch.float32
            The graph regularization loss.
        """
        graph_reg = 0
        if self.smoothness_ratio not in (0, None):
            for i in range(adj.shape[0]):
                L = torch.diagflat(torch.sum(adj[i], -1)) - adj[i]
                graph_reg += (
                    self.smoothness_ratio
                    * torch.trace(
                        torch.mm(node_feat[i].transpose(-1, -2), torch.mm(L, node_feat[i]))
                    )
                    / int(np.prod(adj.shape))
                )

        if self.connectivity_ratio not in (0, None):
            ones_vec = torch.ones(adj.shape[:-1]).to(adj.device)
            graph_reg += (
                -self.connectivity_ratio
                * torch.matmul(
                    ones_vec.unsqueeze(1),
                    torch.log(torch.matmul(adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER),
                ).sum()
                / adj.shape[0]
                / adj.shape[-1]
            )

        if self.sparsity_ratio not in (0, None):
            graph_reg += (
                self.sparsity_ratio * torch.sum(torch.pow(adj, 2)) / int(np.prod(adj.shape))
            )

        return graph_reg

    def _build_knn_neighbourhood(self, attention, top_k_neigh):
        """Build kNN neighborhood graph.

        Parameters
        ----------
        attention : torch.Tensor
            The attention matrix.
        top_k_neigh : int
            The top k value for kNN neighborhood graph.

        Returns
        -------
        torch.Tensor
            The sparsified adjacency matrix.
        """
        top_k_neigh = min(top_k_neigh, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, top_k_neigh, dim=-1)
        weighted_adj = (
            (self.mask_off_val * torch.ones_like(attention))
            .scatter_(-1, knn_ind, knn_val)
            .to(attention.device)
        )
        weighted_adj[weighted_adj <= 0] = 0
        return weighted_adj

    def _build_epsilon_neighbourhood(self, attention, epsilon_neigh):
        """Build epsilon neighbourhood graph.

        Parameters
        ----------
        attention : torch.Tensor
            The attention matrix.
        epsilon_neigh : float
            The threshold value for epsilon neighbourhood graph.

        Returns
        -------
        torch.Tensor
            The sparsified adjacency matrix.
        """
        mask = (attention > epsilon_neigh).detach().float()
        weighted_adj = attention * mask + self.mask_off_val * (1 - mask)

        return weighted_adj

    def _compute_distance_matrix(self, X, weight=None):
        """Compute distance matrix for RBF kernel.

        Parameters
        ----------
        X : torch.Tensor
            The input node embedding matrix.
        weight : torch.Tensor, optional
            The learnable weight matrix, default ``None``.

        Returns
        -------
        torch.Tensor
            The distance matrix.
        """
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X

        norm = torch.sum(trans_X * X, dim=-1)
        dists = (
            -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        )

        return dists

    def _get_node_mask_for_batch_graph(self, num_nodes):
        """Get node mask matrix for batch graph.

        Parameters
        ----------
        num_nodes : torch.Tensor
            The node size matrix.

        Returns
        -------
        torch.Tensor
            The node mask matrix.
        """
        node_mask = []
        for i in range(num_nodes.shape[0]):  # batch
            graph_node_num = num_nodes[i].item()
            node_mask.append(sparse.coo_matrix(np.ones((graph_node_num, graph_node_num))))

        node_mask = sparse.block_diag(node_mask)
        node_mask = sparse_mx_to_torch_sparse_tensor(node_mask).to_dense().to(num_nodes.device)

        return node_mask

    def _get_normalized_init_adj(self, graph):
        """Compute the symmetric normalized Laplacian matrix of the input graph data.

        Parameters
        ----------
        graph : GraphData
            The input graph data.

        Returns
        -------
        torch.Tensor
            The symmetric normalized Laplacian matrix.
        """
        norm_init_adj = graph.adj_matrix(batch_view=True, post_processing_fn=normalize_adj)

        return norm_init_adj
