import numpy as np
from scipy import sparse
import torch
from torch import nn
import torch.nn.functional as F

from .embedding_construction import EmbeddingConstruction
from ..utils.constants import INF
from ..utils.generic_utils import normalize_sparse_adj, sparse_mx_to_torch_sparse_tensor, to_cuda
from ..utils.constants import VERY_SMALL_NUMBER


class GraphConstructionBase(nn.Module):
    """Base class for graph construction.

    Parameters
    ----------
    word_vocab : Vocab
        The word vocabulary.
    embedding_styles : dict
        - ``single_token_item`` : specify whether the item (i.e., node or edge) contains single token or multiple tokens.
        - ``emb_strategy`` : specify the embedding construction strategy.
        - ``num_rnn_layers``: specify the number of RNN layers.
        - ``bert_model_name``: specify the BERT model name.
        - ``bert_lower_case``: specify whether to lower case the input text for BERT embeddings.
    hidden_size : int, optional
        The hidden size of RNN layer, default: ``None``.
    fix_word_emb : boolean, optional
        Specify whether to fix pretrained word embeddings, default: ``True``.
    fix_bert_emb : boolean, optional
        Specify whether to fix pretrained BERT embeddings, default: ``True``.
    bert_model_name : str, optional
        Specify the BERT model name, default: ``'bert-base-uncased'``.
    bert_lower_case : bool, optional
        Specify whether to lower case the input text for BERT embeddings, default: ``True``.
    word_dropout : float, optional
        Dropout ratio for word embedding, default: ``None``.
    rnn_dropout : float, optional
        Dropout ratio for RNN embedding, default: ``None``.
    device : torch.device, optional
        Specify computation device (e.g., CPU), default: ``None`` for using CPU.
    """
    def __init__(self,
                word_vocab,
                embedding_styles,
                hidden_size=None,
                fix_word_emb=True,
                fix_bert_emb=True,
                word_dropout=None,
                rnn_dropout=None,
                device=None):
        super(GraphConstructionBase, self).__init__()
        self.embedding_layer = EmbeddingConstruction(word_vocab,
                                        embedding_styles['single_token_item'],
                                        emb_strategy=embedding_styles['emb_strategy'],
                                        hidden_size=hidden_size,
                                        num_rnn_layers=embedding_styles.get('num_rnn_layers', 1),
                                        fix_word_emb=fix_word_emb,
                                        fix_bert_emb=fix_bert_emb,
                                        bert_model_name=embedding_styles.get('bert_model_name', 'bert-base-uncased'),
                                        bert_lower_case=embedding_styles.get('bert_lower_case', True),
                                        word_dropout=word_dropout,
                                        rnn_dropout=rnn_dropout,
                                        device=device)

    def forward(self, raw_text_data, **kwargs):
        """Compute graph topology and initial node/edge embeddings.

        Parameters
        ----------
        raw_text_data :
            The raw text data.
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

    def topology(self, **kwargs):
        """Compute graph topology.

        Parameters
        ----------
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

    def embedding(self, **kwargs):
        """Compute initial node/edge embeddings.

        Parameters
        ----------
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

class StaticGraphConstructionBase(GraphConstructionBase):
    """
    Base class for static graph construction.

    ...

    Attributes
    ----------
    embedding_styles : dict
        Specify embedding styles including ``word_emb_type``, ``node_edge_emb_strategy`` and ``seq_info_encode_strategy``.

    Methods
    -------
    add_vocab()
        Add new parsed words or syntactic components into vocab.

    topology()
        Generate graph topology.

    embedding(raw_data, structure)
        Generate graph embeddings.

    forward(raw_data)
        Generate static graph embeddings and topology.
    """

    def __init__(self, word_vocab, embedding_styles, hidden_size,
                 fix_word_emb=True, fix_bert_emb=True, word_dropout=None, rnn_dropout=None, device=None):
        super(StaticGraphConstructionBase, self).__init__(word_vocab,
                                                           embedding_styles,
                                                           hidden_size,
                                                           fix_word_emb=fix_word_emb,
                                                           fix_bert_emb=fix_bert_emb,
                                                           word_dropout=word_dropout,
                                                           rnn_dropout=rnn_dropout,
                                                           device=device)

    def add_vocab(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    def parsing(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    def topology(cls, **kwargs):
        raise NotImplementedError()

    def embedding(self, **kwargs):
        raise NotImplementedError()

    def forward(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _construct_static_graph(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _graph_connect(cls, **kwargs):
        raise NotImplementedError()

class DynamicGraphConstructionBase(GraphConstructionBase):
    """Base class for dynamic graph construction.

    Parameters
    ----------
    word_vocab : Vocab
        The word vocabulary.
    embedding_styles : dict
        - ``single_token_item`` : specify whether the item (i.e., node or edge) contains single token or multiple tokens.
        - ``emb_strategy`` : specify the embedding construction strategy.
        - ``num_rnn_layers``: specify the number of RNN layers.
        - ``bert_model_name``: specify the BERT model name.
        - ``bert_lower_case``: specify whether to lower case the input text for BERT embeddings.
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
    fix_word_emb : boolean, optional
        Specify whether to fix pretrained word embeddings, default: ``False``.
    fix_bert_emb : boolean, optional
        Specify whether to fix pretrained BERT embeddings, default: ``True``.
    bert_model_name : str, optional
        Specify the BERT model name, default: ``'bert-base-uncased'``.
    bert_lower_case : bool, optional
        Specify whether to lower case the input text for BERT embeddings, default: ``True``.
    word_dropout : float, optional
        Dropout ratio for word embedding, default: ``None``.
    rnn_dropout : float, optional
        Dropout ratio for RNN embedding, default: ``None``.
    device : torch.device, optional
        Specify computation device (e.g., CPU), default: ``None`` for using CPU.
    """
    def __init__(self,
                word_vocab,
                embedding_styles,
                sim_metric_type='weighted_cosine',
                num_heads=1,
                top_k_neigh=None,
                epsilon_neigh=None,
                smoothness_ratio=None,
                connectivity_ratio=None,
                sparsity_ratio=None,
                input_size=None,
                hidden_size=None,
                fix_word_emb=False,
                fix_bert_emb=False,
                word_dropout=None,
                rnn_dropout=None,
                device=None):
        super(DynamicGraphConstructionBase, self).__init__(
                                                    word_vocab,
                                                    embedding_styles,
                                                    hidden_size=hidden_size,
                                                    fix_word_emb=fix_word_emb,
                                                    fix_bert_emb=fix_bert_emb,
                                                    word_dropout=word_dropout,
                                                    rnn_dropout=rnn_dropout,
                                                    device=device)
        assert top_k_neigh is None or epsilon_neigh is None, \
            'top_k_neigh and epsilon_neigh cannot be activated at the same time!'
        self.device = device
        self.top_k_neigh = top_k_neigh
        self.epsilon_neigh = epsilon_neigh
        self.sim_metric_type = sim_metric_type
        self.smoothness_ratio = smoothness_ratio
        self.connectivity_ratio = connectivity_ratio
        self.sparsity_ratio = sparsity_ratio

        if self.sim_metric_type == 'attention':
            self.mask_off_val = -INF
            self.linear_sims = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False)
                                                    for _ in range(num_heads)])
        elif self.sim_metric_type == 'weighted_cosine':
            self.mask_off_val = 0
            self.weight = torch.Tensor(num_heads, input_size)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        elif self.sim_metric_type == 'gat_attention':
            self.mask_off_val = -INF
            self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False)
                                                    for _ in range(num_heads)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False)
                                                    for _ in range(num_heads)])
            self.leakyrelu = nn.LeakyReLU(0.2)
        elif self.sim_metric_type == 'rbf_kernel':
            self.mask_off_val = 0
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        elif self.sim_metric_type == 'cosine':
            self.mask_off_val = 0
        else:
            raise RuntimeError('Unknown sim_metric_type: {}'.format(self.sim_metric_type))


    def forward(self, raw_text_data, **kwargs):
        """Compute graph topology and initial node/edge embeddings.

        Parameters
        ----------
        raw_text_data : list of sequences.
            The raw text data.
        **kwargs
            Extra parameters.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

    def topology(self, node_emb, edge_emb=None,
                    init_adj=None, node_mask=None, **kwargs):
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

    def embedding(self, feat, **kwargs):
        """Compute initial node/edge embeddings.

        Parameters
        ----------
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
        if self.sim_metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                node_vec_t = torch.relu(self.linear_sims[_](node_emb))
                attention += torch.matmul(node_vec_t, node_vec_t.transpose(-1, -2))

            attention /= len(self.linear_sims)
        elif self.sim_metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight.unsqueeze(1)
            if len(node_emb.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            node_vec_t = node_emb.unsqueeze(0) * expand_weight_tensor
            node_vec_norm = F.normalize(node_vec_t, p=2, dim=-1)
            attention = torch.matmul(node_vec_norm, node_vec_norm.transpose(-1, -2)).mean(0)
        elif self.sim_metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](node_emb)
                a_input2 = self.linear_sims2[_](node_emb)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
        elif self.sim_metric_type == 'rbf_kernel':
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self._compute_distance_matrix(node_emb, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis**2))
        elif self.sim_metric_type == 'cosine':
            node_vec_norm = node_emb.div(torch.norm(node_emb, p=2, dim=-1, keepdim=True))
            attention = torch.mm(node_vec_norm, node_vec_norm.transpose(-1, -2)).detach()

        if node_mask is not None:
            if torch.__version__ < '1.3.0':
                attention = attention.masked_fill_(~(node_mask == 1.), self.mask_off_val)
            else:
                attention = attention.masked_fill_(~node_mask.bool(), self.mask_off_val)


        return attention

    def sparsify_graph(self, adj):
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
        if not self.smoothness_ratio in (0, None):
            L = torch.diagflat(torch.sum(adj, -1)) - adj
            graph_reg += self.smoothness_ratio / int(np.prod(adj.shape))\
                    * torch.trace(torch.mm(node_feat.transpose(-1, -2), torch.mm(L, node_feat)))

        if not self.connectivity_ratio in (0, None):
            ones_vec = to_cuda(torch.ones(adj.size(-1)), self.device)
            graph_reg += -self.connectivity_ratio / adj.shape[-1]\
                    * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).squeeze()

        if not self.sparsity_ratio in (0, None):
            graph_reg += self.sparsity_ratio / int(np.prod(adj.shape))\
                    * torch.sum(torch.pow(adj, 2))

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
        weighted_adj = to_cuda((self.mask_off_val * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), self.device)

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
        """
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X

        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)

        return dists

    def _get_node_mask_for_batch_graph(self, num_nodes):
        node_mask = []
        for i in range(num_nodes.shape[0]): # batch
            graph_node_num = num_nodes[i].item()
            node_mask.append(sparse.coo_matrix(np.ones((graph_node_num, graph_node_num))))

        node_mask = sparse.block_diag(node_mask)
        node_mask = to_cuda(sparse_mx_to_torch_sparse_tensor(node_mask).to_dense(), self.device)

        return node_mask

    def _get_normalized_init_adj(self, graph):
        adj = graph.scipy_sparse_adj()
        adj = normalize_sparse_adj(adj)
        adj = to_cuda(sparse_mx_to_torch_sparse_tensor(adj), self.device)

        return adj
