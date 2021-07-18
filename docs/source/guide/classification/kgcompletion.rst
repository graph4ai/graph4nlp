.. _guide-kgcompletion:

Knowledge Graph Completion
=====================
The purpose of Knowledge Graph Completion (KGC) is to predict new triples on the basis of existing triples,
so as to further extend KGs. KGC is usually considered as a link prediction task.
Formally, the knowledge graph is represented by :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})`,
in which entities :math:`v_i \in \mathcal{V}`, edges :math:`(v_s, r, v_o) \in \mathcal{E}`,
and :math:`r \in \mathcal{R}` is a relation type. This task scores for new facts
(i.e. triples like :math:`\left \langle subject, relation, object \right \rangle`) to
determine how likely those edges are to belong to :math:`\mathcal{E}`.

KGC can be solved with an encoder-decoder framework. To encode the local neighborhood
information of an entity, the encoder can be chosen from a variety of GNNs.

The decoder is a knowledge graph embedding model and can be regarded as a scoring
function. The most common decoders of knowledge graph completion includes
translation-based models (TransE), tensor factorization based models (DistMult,
ComplEx) and neural network base models (ConvE).
We implement ``DistMult`` and ``ComplEx`` in this library.

DistMult
--------
DistMult is a tensor factorization based models from paper `Embedding entities and
relations for learning and inference in knowledge bases <https://arxiv.org/pdf/1412.6575.pdf>`__.
For ``DistMult``, the equation is:

.. math::

    f(s, r, o) = e_s^T M_r e_o,
    e_s, e_o \in \mathbb{R}^d,
    M_r \in \mathbb{R}^{d \times d}

In DistMult, every relation r is represented by a diagonal matrix :math:`M_r \in \mathbb{R}^{d \times d}`
and a triple is scored as :math:`f(s, r, o) = e_s^T M_r e_o`.

In our implementation, the subject embedding, relation embedding and all entity
embeddings are given as the ``forward(...)`` input. Then, we compute the score logits
for all entity nodes using multi-class loss such as ``BCELoss()`` or the predition
scores of positive/negative examples using pairwise Loss Function such as ``SoftplusLoss()``
and ``SigmoidLoss()``. More details about the KG completion loss please refer to :ref:`graph4nlp.loss.KGLoss`.

.. code::

    class DistMult(KGCompletionBase):

        def __init__(self,
                     input_dropout=0.0,
                     loss_name='BCELoss'):
            super(DistMult, self).__init__()
            self.loss_name = loss_name
            self.classifier = DistMultLayer(input_dropout, loss_name)

        def forward(self, input_graph: GraphData, e1_emb, rel_emb, all_node_emb, multi_label=None):

            if multi_label is None:
                input_graph.graph_attributes['logits'] = self.classifier(e1_emb,
                                                                         rel_emb,
                                                                         all_node_emb)  # [B, N]
            else:
                input_graph.graph_attributes['logits'], input_graph.graph_attributes['p_score'], \
                input_graph.graph_attributes['n_score'] = self.classifier(e1_emb,
                                                                          rel_emb,
                                                                          all_node_emb,
                                                                          multi_label)
                # input_graph.graph_attributes['p_score']: [L_p]
                # input_graph.graph_attributes['n_score']: [L_n]
                # L_p + L_n == B * N

            return input_graph


    class DistMultLayer(KGCompletionLayerBase):
        def __init__(self,
                     input_dropout=0.0,
                     loss_name='BCELoss'):
            super(DistMultLayer, self).__init__()
            self.inp_drop = nn.Dropout(input_dropout)
            self.loss_name = loss_name

        def forward(self,
                    e1_emb,
                    rel_emb,
                    all_node_emb,
                    multi_label=None):

            # dropout
            e1_emb = self.inp_drop(e1_emb)
            rel_emb = self.inp_drop(rel_emb)

            logits = torch.mm(e1_emb * rel_emb,
                              all_node_emb.weight.transpose(1, 0))


            if self.loss_name in ['SoftMarginLoss']:
                # target labels are numbers selecting from -1 and 1.
                pred = torch.tanh(logits)
            else:
                # target labels are numbers selecting from 0 and 1.
                pred = torch.sigmoid(logits)

            if multi_label is not None:
                idxs_pos = torch.nonzero(multi_label == 1.)
                pred_pos = pred[idxs_pos[:, 0], idxs_pos[:, 1]]

                idxs_neg = torch.nonzero(multi_label == 0.)
                pred_neg = pred[idxs_neg[:, 0], idxs_neg[:, 1]]
                return pred, pred_pos, pred_neg
            else:
                return pred

ComplEx
-------

ComplEx is proposed in paper `Complex Embeddings for Simple Link Prediction <http://proceedings.mlr.press/v48/trouillon16.pdf>`__.
For ``ComplEx``, the equation is:

.. math::

    \text{Re}(e_r, e_s, \bar{e_t})=\text{Re}(\sum_{k=1}^K e_r e_s\bar{e_t})
    , e_s, e_o \in \mathbb{C}^d
    , e_r \in \mathbb{C}^d

:math:`Re()` denotes the real part of a vector.

How to Combine KGC Decoder with GNN Encoder
-------------------------------------------

The code below provides an end-to-end KGC model using ``GCN`` as encoder and ``DistMult`` as decoder:


.. code:: python

    from graph4nlp.pytorch.modules.graph_embedding.gcn import GCN
    from graph4nlp.pytorch.modules.prediction.classification.kg_completion import DistMult
    from torch.nn.init import xavier_normal_

    class GCNDistMult(torch.nn.Module):
        def __init__(self, args, num_entities, num_relations, num_layers=2):
            super(GCNDistMult, self).__init__()
            self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
            self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)

            self.num_entities = num_entities
            self.num_relations = num_relations

            self.num_layers = num_layers
            self.gnn = GCN(self.num_layers, args.embedding_dim, args.embedding_dim, args.embedding_dim,
                           args.direction_option, feat_drop=args.input_drop)

            self.direction_option = args.direction_option
            self.distmult = DistMult(args.input_drop, loss_name='BCELoss')
            self.loss = torch.nn.BCELoss()

        def init(self):
            xavier_normal_(self.emb_e.weight.data)
            xavier_normal_(self.emb_rel.weight.data)

        def forward(self, e1, rel, kg_graph=None):
            X = torch.LongTensor([i for i in range(self.num_entities)]).to(e1.device)

            kg_graph.node_features['node_feat'] = self.emb_e(X)
            kg_graph = self.gnn(kg_graph)

            e1_embedded = kg_graph.node_features['node_feat'][e1]
            rel_embedded = self.emb_rel(rel)
            e1_embedded = e1_embedded.squeeze()
            rel_embedded = rel_embedded.squeeze()

            kg_graph = self.distmult(kg_graph, e1_embedded, rel_embedded, self.emb_e)
            logits = kg_graph.graph_attributes['logits']

            return logits
