import os
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F

from graph4nlp.pytorch.modules.graph_construction import NodeEmbeddingBasedRefinedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import (
    WordEmbedding,
)
from graph4nlp.pytorch.modules.graph_embedding_initialization.graph_embedding_initialization import (  # noqa
    GraphEmbeddingInitialization,
)
from graph4nlp.pytorch.modules.graph_embedding_learning.gat import GAT
from graph4nlp.pytorch.modules.graph_embedding_learning.gcn import GCN
from graph4nlp.pytorch.modules.graph_embedding_learning.ggnn import GGNN
from graph4nlp.pytorch.modules.graph_embedding_learning.graphsage import GraphSAGE
from graph4nlp.pytorch.modules.prediction.classification.node_classification import (
    BiLSTMFeedForwardNN,
)

torch.multiprocessing.set_sharing_strategy("file_system")
cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# from torchcrf import CRF


def logits2index(logits):
    _, pred = torch.max(logits, dim=-1)
    # print(pred.size())
    return pred


class SentenceBiLSTMCRF(nn.Module):
    def __init__(self, args, device=None, use_rnn=False):
        super(SentenceBiLSTMCRF, self).__init__()
        self.use_rnn = use_rnn
        # if self.use_rnn is True:
        self.prediction = BiLSTMFeedForwardNN(
            args.init_hidden_size * 1, args.init_hidden_size * 1
        ).to(device)

        # self.crf=CRFLayer(8).to(device)
        # self.use_crf=use_crf
        self.linear1 = nn.Linear(int(args.init_hidden_size * 1), args.hidden_size)
        self.linear1_ = nn.Linear(int(args.hidden_size * 1), args.num_class)
        self.dropout_tag = nn.Dropout(args.tag_dropout)
        self.dropout_rnn_out = nn.Dropout(p=args.rnn_dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss()

    def forward(self, batch_graph, tgt_tags):

        batch_graph = self.prediction(batch_graph)
        batch_emb = batch_graph.node_features["logits"]

        batch_graph.node_features["logits"] = self.linear1_(
            self.dropout_tag(F.elu(self.linear1(self.dropout_rnn_out(batch_emb))))
        )

        logits = batch_graph.node_features["logits"][:, :]  # [batch*sentence*num_nodes,num_lable]
        if tgt_tags is None:
            loss = 0
        else:
            tgt = torch.cat(tgt_tags)
            loss = self.nll_loss(self.logsoftmax(logits), tgt)

        return loss, logits


class Word2tag(nn.Module):
    def __init__(self, vocab, args, device=None):
        super(Word2tag, self).__init__()
        self.vocab = vocab
        self.vocab_model = vocab

        self.device = device
        self.graph_name = args.graph_name

        embedding_style = {
            "single_token_item": True if args.graph_name != "ie" else False,
            "emb_strategy": "w2v_bilstm",
            "num_rnn_layers": 1,
            "bert_model_name": "bert-base-uncased",
            "bert_lower_case": True,
        }

        self.graph_initializer = GraphEmbeddingInitialization(
            word_vocab=self.vocab_model.in_word_vocab,
            embedding_style=embedding_style,
            hidden_size=args.init_hidden_size,
            word_dropout=args.word_dropout,
            rnn_dropout=None,
            fix_word_emb=not args.no_fix_word_emb,
            fix_bert_emb=not args.no_fix_word_emb,
        )

        if args.graph_name == "node_emb":
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(
                sim_metric_type=args.gl_metric_type,
                num_heads=args.gl_num_heads,
                top_k_neigh=args.gl_top_k,
                epsilon_neigh=args.gl_epsilon,
                smoothness_ratio=args.gl_smoothness_ratio,
                connectivity_ratio=args.gl_connectivity_ratio,
                sparsity_ratio=args.gl_sparsity_ratio,
                input_size=args.init_hidden_size,
                hidden_size=args.init_hidden_size,
                fix_word_emb=not args.no_fix_word_emb,
                fix_bert_emb=not args.no_fix_bert_emb,
                word_dropout=args.word_dropout,
                rnn_dropout=None,
                device=self.device,
                use_edge_weight=True,
            )

        if args.graph_name == "node_emb_refined":
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
                args.init_adj_alpha,
                sim_metric_type=args.gl_metric_type,
                num_heads=args.gl_num_heads,
                top_k_neigh=args.gl_top_k,
                epsilon_neigh=args.gl_epsilon,
                smoothness_ratio=args.gl_smoothness_ratio,
                connectivity_ratio=args.gl_connectivity_ratio,
                sparsity_ratio=args.gl_sparsity_ratio,
                input_size=args.init_hidden_size,
                hidden_size=args.init_hidden_size,
                fix_word_emb=not args.no_fix_word_emb,
                word_dropout=args.word_dropout,
                rnn_dropout=None,
                device=self.device,
                use_edge_weight=True,
            )

        if "w2v" in self.graph_initializer.embedding_layer.word_emb_layers:
            self.word_emb = self.graph_initializer.embedding_layer.word_emb_layers[
                "w2v"
            ].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                self.vocab.in_word_vocab.embeddings.shape[0],
                self.vocab.in_word_vocab.embeddings.shape[1],
                pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                fix_emb=not args.no_fix_word_emb,
                device=self.device,
            ).word_emb_layer

        self.gnn_type = args.gnn_type
        self.use_gnn = args.use_gnn
        self.linear0 = nn.Linear(int(args.init_hidden_size * 1), args.hidden_size).to(self.device)
        self.linear0_ = nn.Linear(int(args.init_hidden_size * 1), args.init_hidden_size).to(
            self.device
        )
        self.dropout_tag = nn.Dropout(args.tag_dropout)
        self.dropout_rnn_out = nn.Dropout(p=args.rnn_dropout)
        if self.use_gnn is False:
            self.bilstmcrf = SentenceBiLSTMCRF(args, device=self.device, use_rnn=False).to(
                self.device
            )
        else:
            if self.gnn_type == "graphsage":
                if args.direction_option == "bi_sep":
                    self.gnn = GraphSAGE(
                        args.gnn_num_layers,
                        args.hidden_size,
                        int(args.init_hidden_size / 2),
                        int(args.init_hidden_size / 2),
                        aggregator_type="mean",
                        direction_option=args.direction_option,
                        activation=F.elu,
                    ).to(self.device)
                else:
                    self.gnn = GraphSAGE(
                        args.gnn_num_layers,
                        args.hidden_size,
                        args.init_hidden_size,
                        args.init_hidden_size,
                        aggregator_type="mean",
                        direction_option=args.direction_option,
                        activation=F.elu,
                    ).to(self.device)
                self.bilstmcrf = SentenceBiLSTMCRF(args, device=self.device, use_rnn=True).to(
                    self.device
                )
            elif self.gnn_type == "ggnn":
                if args.direction_option == "bi_sep":
                    self.gnn = GGNN(
                        args.gnn_num_layers,
                        int(args.init_hidden_size / 2),
                        int(args.init_hidden_size / 2),
                        direction_option=args.direction_option,
                        n_etypes=1,
                    ).to(self.device)
                else:
                    self.gnn = GGNN(
                        args.gnn_num_layers,
                        args.init_hidden_size,
                        args.init_hidden_size,
                        direction_option=args.direction_option,
                        n_etypes=1,
                    ).to(self.device)
                self.bilstmcrf = SentenceBiLSTMCRF(args, device=self.device, use_rnn=True).to(
                    self.device
                )
            elif self.gnn_type == "gat":
                heads = 2
                if args.direction_option == "bi_sep":
                    self.gnn = GAT(
                        args.gnn_num_layers,
                        args.hidden_size,
                        int(args.init_hidden_size / 2),
                        int(args.init_hidden_size / 2),
                        heads,
                        direction_option=args.direction_option,
                        feat_drop=0.6,
                        attn_drop=0.6,
                        negative_slope=0.2,
                        activation=F.elu,
                    ).to(self.device)
                else:
                    self.gnn = GAT(
                        args.gnn_num_layers,
                        args.hidden_size,
                        args.init_hidden_size,
                        args.init_hidden_size,
                        heads,
                        direction_option=args.direction_option,
                        feat_drop=0.6,
                        attn_drop=0.6,
                        negative_slope=0.2,
                        activation=F.elu,
                    ).to(self.device)
                self.bilstmcrf = SentenceBiLSTMCRF(args, device=self.device, use_rnn=True).to(
                    self.device
                )
            elif self.gnn_type == "gcn":
                if args.direction_option == "bi_sep":
                    self.gnn = GCN(
                        args.gnn_num_layers,
                        args.hidden_size,
                        int(args.init_hidden_size / 2),
                        int(args.init_hidden_size / 2),
                        direction_option=args.direction_option,
                        activation=F.elu,
                    ).to(self.device)
                else:
                    self.gnn = GCN(
                        args.gnn_num_layers,
                        args.hidden_size,
                        args.init_hidden_size,
                        args.init_hidden_size,
                        direction_option=args.direction_option,
                        activation=F.elu,
                    ).to(self.device)
                self.bilstmcrf = SentenceBiLSTMCRF(args, device=self.device, use_rnn=True).to(
                    self.device
                )

    def forward(self, graph, tgt=None, require_loss=True):
        batch_graph = self.graph_initializer(graph)

        if self.use_gnn is False:
            batch_graph.node_features["node_emb"] = batch_graph.node_features["node_feat"]
            batch_graph.node_features["node_emb"] = self.dropout_tag(
                F.elu(self.linear0_(self.dropout_rnn_out(batch_graph.node_features["node_emb"])))
            )

        else:
            # run GNN
            if self.gnn_type == "ggnn":
                batch_graph.node_features["node_feat"] = batch_graph.node_features["node_feat"]
            else:
                batch_graph.node_features["node_feat"] = self.dropout_tag(
                    F.elu(
                        self.linear0(self.dropout_rnn_out(batch_graph.node_features["node_feat"]))
                    )
                )

            batch_graph = self.gnn(batch_graph)

        # down-task
        loss, logits = self.bilstmcrf(batch_graph, tgt)
        pred_index = logits2index(logits)
        self.loss = loss

        if require_loss is True:
            return pred_index, self.loss
        else:
            loss = None
            return pred_index, self.loss

    def inference_forward(self, collate_data):
        batch_graph = collate_data["graph_data"].to(self.device)
        return self.forward(batch_graph, tgt=None, require_loss=False)[0]

    def post_process(self, logits, label_names):
        logits_list = []
        # tgt_list = []

        for idx in range(len(logits)):
            logits_list.append(logits[idx].cpu().clone().numpy())
        pred_tags = [label_names[int(pred)] for pred in logits_list]
        return pred_tags

    def save_checkpoint(self, save_path, checkpoint_name):
        """
            The API for saving the model.
        Parameters
        ----------
        save_path : str
            The root path.
        checkpoint_name : str
            The name of the checkpoint.
        Returns
        -------

        """
        checkpoint_path = os.path.join(save_path, checkpoint_name)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self, checkpoint_path)

    @classmethod
    def load_checkpoint(cls, load_path, checkpoint_name):
        """
            The API to load the model.

        Parameters
        ----------
        load_path : str
            The root path to load the model.
        checkpoint_name : str
            The name of the model to be loaded.

        Returns
        -------
        Graph2XBase
        """
        checkpoint_path = os.path.join(load_path, checkpoint_name)
        model = torch.load(checkpoint_path)
        return model
