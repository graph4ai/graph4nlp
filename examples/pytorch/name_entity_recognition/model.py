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
    def __init__(self, config, device=None, use_rnn=False):
        super(SentenceBiLSTMCRF, self).__init__()
        self.use_rnn = use_rnn
        # if self.use_rnn is True:
        if (
            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                "direction_option"
            ]
            != "bi_sep"
            and config["model_args"]["graph_embedding_name"] == "graphsage"
        ):
            self.prediction = BiLSTMFeedForwardNN(
                int(config["model_args"]["init_hidden_size"] / 2),
                int(config["model_args"]["init_hidden_size"] / 2),
            ).to(device)
            self.linear1 = nn.Linear(
                int(config["model_args"]["init_hidden_size"] / 2),
                config["model_args"]["hidden_size"],
            )
            self.linear1_ = nn.Linear(
                int(config["model_args"]["hidden_size"] * 1), config["model_args"]["num_class"]
            )
        else:
            self.prediction = BiLSTMFeedForwardNN(
                config["model_args"]["init_hidden_size"] * 1,
                config["model_args"]["init_hidden_size"] * 1,
            ).to(device)
            self.linear1 = nn.Linear(
                int(config["model_args"]["init_hidden_size"] * 1),
                config["model_args"]["hidden_size"],
            )
            self.linear1_ = nn.Linear(
                int(config["model_args"]["hidden_size"] * 1), config["model_args"]["num_class"]
            )
        # self.crf=CRFLayer(8).to(device)
        # self.use_crf=use_crf

        self.dropout_tag = nn.Dropout(config["model_args"]["tag_dropout"])
        self.dropout_rnn_out = nn.Dropout(p=config["model_args"]["rnn_dropout"])
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
    def __init__(self, vocab, config, device=None):
        super(Word2tag, self).__init__()
        self.vocab = vocab
        self.vocab_model = vocab
        self.config = config
        self.device = device
        self.graph_name = self.config["model_args"]["graph_construction_name"]
        self.graph_construction_name = self.graph_name

        # embedding_style = {
        #    "single_token_item": True if self.graph_name != "ie" else False,
        #    "emb_strategy": "w2v_bilstm",
        #    "num_rnn_layers": 1,
        #    "bert_model_name": "bert-base-uncased",
        #    "bert_lower_case": True,
        # }

        self.graph_initializer = GraphEmbeddingInitialization(
            word_vocab=self.vocab_model.in_word_vocab,
            embedding_style=config["model_args"]["graph_initialization_args"]["embedding_style"],
            hidden_size=config["model_args"]["graph_initialization_args"]["hidden_size"],
            word_dropout=config["model_args"]["graph_initialization_args"]["word_dropout"],
            rnn_dropout=config["model_args"]["graph_initialization_args"]["rnn_dropout"],
            fix_word_emb=config["model_args"]["graph_initialization_args"]["rnn_dropout"],
            fix_bert_emb=config["model_args"]["graph_initialization_args"]["fix_bert_emb"],
        )

        if self.graph_name == "node_emb":
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(
                sim_metric_type=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["sim_metric_type"],
                num_heads=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["num_heads"],
                top_k_neigh=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["top_k_neigh"],
                epsilon_neigh=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["epsilon_neigh"],
                smoothness_ratio=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["smoothness_ratio"],
                connectivity_ratio=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["connectivity_ratio"],
                sparsity_ratio=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["sparsity_ratio"],
                input_size=config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "input_size"
                ],
                hidden_size=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["hidden_size"],
            )

        if self.graph_name == "node_emb_refined":
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
                config["model_args"]["graph_construction_args"]["graph_construction_private"][
                    "alpha_fusion"
                ],
                sim_metric_type=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["sim_metric_type"],
                num_heads=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["num_heads"],
                top_k_neigh=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["top_k_neigh"],
                epsilon_neigh=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["epsilon_neigh"],
                smoothness_ratio=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["smoothness_ratio"],
                connectivity_ratio=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["connectivity_ratio"],
                sparsity_ratio=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["sparsity_ratio"],
                input_size=config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "input_size"
                ],
                hidden_size=config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ]["hidden_size"],
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
                fix_emb=config["graph_initialization_args"]["fix_word_emb"],
                device=self.device,
            ).word_emb_layer

        self.gnn_type = config["model_args"]["graph_embedding_name"]
        self.use_gnn = config["model_args"]["use_gnn"]
        self.linear0 = nn.Linear(
            int(config["model_args"]["init_hidden_size"] * 1), config["model_args"]["hidden_size"]
        ).to(self.device)
        self.linear0_ = nn.Linear(
            int(config["model_args"]["init_hidden_size"] * 1),
            config["model_args"]["init_hidden_size"],
        ).to(self.device)
        self.dropout_tag = nn.Dropout(config["model_args"]["tag_dropout"])
        self.dropout_rnn_out = nn.Dropout(p=config["model_args"]["rnn_dropout"])
        if self.use_gnn is False:
            self.bilstmcrf = SentenceBiLSTMCRF(config, device=self.device, use_rnn=False).to(
                self.device
            )
        else:
            if self.gnn_type == "graphsage":
                if (
                    config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                        "direction_option"
                    ]
                    == "bi_sep"
                ):
                    self.gnn = GraphSAGE(
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "gnn_num_layers"
                        ],
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "hidden_size"
                        ],
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                            / 2
                        ),
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                            / 2
                        ),
                        aggregator_type="mean",
                        direction_option=config["model_args"]["graph_embedding_args"][
                            "graph_embedding_share"
                        ]["direction_option"],
                        activation=F.elu,
                    ).to(self.device)
                else:
                    self.gnn = GraphSAGE(
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "gnn_num_layers"
                        ],
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "hidden_size"
                        ],
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                            / 2
                        ),
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                            / 2
                        ),
                        aggregator_type="mean",
                        direction_option=config["model_args"]["graph_embedding_args"][
                            "graph_embedding_share"
                        ]["direction_option"],
                        activation=F.elu,
                    ).to(self.device)
                self.bilstmcrf = SentenceBiLSTMCRF(config, device=self.device, use_rnn=True).to(
                    self.device
                )
            elif self.gnn_type == "ggnn":
                if (
                    config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                        "direction_option"
                    ]
                    == "bi_sep"
                ):
                    self.gnn = GGNN(
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "gnn_num_layers"
                        ],
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                            / 2
                        ),
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                            / 2
                        ),
                        direction_option=config["model_args"]["graph_embedding_args"][
                            "graph_embedding_share"
                        ]["direction_option"],
                        n_etypes=1,
                    ).to(self.device)
                else:
                    self.gnn = GGNN(
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "gnn_num_layers"
                        ],
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "hidden_size"
                        ],
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                        ),
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                        ),
                        direction_option=config["model_args"]["graph_embedding_args"][
                            "graph_embedding_share"
                        ]["direction_option"],
                        n_etypes=1,
                    ).to(self.device)
                self.bilstmcrf = SentenceBiLSTMCRF(config, device=self.device, use_rnn=True).to(
                    self.device
                )
            elif self.gnn_type == "gat":
                heads = 2
                if (
                    config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                        "direction_option"
                    ]
                    == "bi_sep"
                ):
                    self.gnn = GAT(
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "gnn_num_layers"
                        ],
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "hidden_size"
                        ],
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                            / 2
                        ),
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                            / 2
                        ),
                        heads,
                        direction_option=config["model_args"]["graph_embedding_args"][
                            "graph_embedding_share"
                        ]["direction_option"],
                        feat_drop=0.6,
                        attn_drop=0.6,
                        negative_slope=0.2,
                        activation=F.elu,
                    ).to(self.device)
                else:
                    self.gnn = GAT(
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "gnn_num_layers"
                        ],
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "hidden_size"
                        ],
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                        ),
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                        ),
                        heads,
                        direction_option=config["model_args"]["graph_embedding_args"][
                            "graph_embedding_share"
                        ]["direction_option"],
                        feat_drop=0.6,
                        attn_drop=0.6,
                        negative_slope=0.2,
                        activation=F.elu,
                    ).to(self.device)
                self.bilstmcrf = SentenceBiLSTMCRF(config, device=self.device, use_rnn=True).to(
                    self.device
                )
            elif self.gnn_type == "gcn":
                if (
                    config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                        "direction_option"
                    ]
                    == "bi_sep"
                ):
                    self.gnn = GCN(
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "gnn_num_layers"
                        ],
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "hidden_size"
                        ],
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                            / 2
                        ),
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                            / 2
                        ),
                        direction_option=config["model_args"]["graph_embedding_args"][
                            "graph_embedding_share"
                        ]["direction_option"],
                        activation=F.elu,
                    ).to(self.device)
                else:
                    self.gnn = GCN(
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "gnn_num_layers"
                        ],
                        config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                            "hidden_size"
                        ],
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                        ),
                        int(
                            config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                                "init_hidden_size"
                            ]
                        ),
                        direction_option=config["model_args"]["graph_embedding_args"][
                            "graph_embedding_share"
                        ]["direction_option"],
                        activation=F.elu,
                    ).to(self.device)
                self.bilstmcrf = SentenceBiLSTMCRF(config, device=self.device, use_rnn=True).to(
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
