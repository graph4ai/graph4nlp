from ..base import KGCompletionBase
from .TransELayer import TransELayer
from .....data.data import GraphData


class TransE(KGCompletionBase):
    r"""Specific class for knowledge graph completion task.
    TransE from paper `Translating Embeddings for Modeling
    Multi-relational Data <https://papers.nips.cc/paper/5071
    -translating-embeddings-for-modeling-multi-relational-data.pdf>`__.

    .. math::
        f(s, r, o) & = ||e_s + w_r - e_o||_p

    Parameters
    ----------
    p_norm: int
        Default: 1

    rel_emb_from_gnn: bool
        If `rel_emb` is computed from GNN, rel_emb_from_gnn is set to `True`.
        Else, rel_emb is initialized as nn.Embedding randomly. Default: `True`.

    num_relations: int
        Number of relations. `num_relations` is needed if rel_emb_from_gnn==True.
        Default: `None`.

    embedding_dim: int
        Dimension of the rel_emb. `embedding_dim` is needed if rel_emb_from_gnn==True.
        Default: `0`.

    edge2node: bool
        The edges or relations in KG are converted to nodes. Defauly: `False`.

    loss_name: str
        The loss type selected fot the KG completion task. Default: `'BCELoss'`
    """

    def __init__(self,
                 p_norm=1,
                 rel_emb_from_gnn=True,
                 num_relations=None,
                 embedding_dim=None,
                 edge2node=False,
                 loss_name='BCELoss'):
        super(TransE, self).__init__()
        self.rel_emb_from_gnn = rel_emb_from_gnn
        self.edge2node = edge2node
        self.classifier = TransELayer(p_norm, rel_emb_from_gnn,
                                       num_relations, embedding_dim, loss_name)


    def forward(self, input_graph: GraphData):
        r"""
        Forward functions to compute the logits tensor for kg completion.

        Parameters
        ----------

        input graph : GraphData
                     The tensors stored in the node feature field named "node_emb" and
                     "rel_emb" in the input_graph are used for knowledge graph completion.

        Returns
        ---------

        output_graph : GraphData
                      The computed logit tensor for each nodes in the graph are stored
                      in the node feature field named "node_logits".
                      logit tensor shape is: [num_class]
        """

        node_emb = input_graph.node_features['node_emb']
        if self.loss_name in ['SoftplusLoss', 'SigmoidLoss']:
            multi_label = input_graph.graph_attributes['multi_binary_label']
        else:
            multi_label = None

        if self.edge2node:
            rel_emb = node_emb
        else:
            if input_graph.edge_features['edge_emb'] != None:
                rel_emb = input_graph.edge_features['edge_emb']
            else:
                assert self.rel_emb_from_gnn == False
                rel_emb = None

        if 'list_e_r_pair_idx' in input_graph.graph_attributes.keys():
            list_e_r_pair_idx = input_graph.graph_attributes['list_e_r_pair_idx']
            list_e_e_pair_idx = None
        elif 'list_e_e_pair_idx' in input_graph.graph_attributes.keys():
            list_e_e_pair_idx = input_graph.graph_attributes['list_e_e_pair_idx']
            list_e_r_pair_idx = None
        else:
            raise RuntimeError("'list_e_r_pair_idx' or 'list_e_e_pair_idx' should be given.")

        if multi_label is None:
            input_graph.graph_attributes['logits'] = self.classifier(node_emb,
                                                                  rel_emb,
                                                                  list_e_r_pair_idx,
                                                                  list_e_e_pair_idx)
        else:
            input_graph.graph_attributes['logits'], input_graph.graph_attributes['p_score'], \
            input_graph.graph_attributes['n_score'] = self.classifier(node_emb,
                                                                   rel_emb,
                                                                   list_e_r_pair_idx,
                                                                   list_e_e_pair_idx,
                                                                   multi_label)

        return input_graph