import argparse
import copy
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.trec import TrecDataset
from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy
from graph4nlp.pytorch.modules.graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
    NodeEmbeddingBasedRefinedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import (
    WordEmbedding,
)
from graph4nlp.pytorch.modules.graph_embedding_initialization.graph_embedding_initialization import (  # noqa
    GraphEmbeddingInitialization,
)
from graph4nlp.pytorch.modules.graph_embedding_learning import GAT, GGNN, GraphSAGE
from graph4nlp.pytorch.modules.graph_embedding_learning.rgcn import RGCN
from graph4nlp.pytorch.modules.loss.general_loss import GeneralLoss
from graph4nlp.pytorch.modules.prediction.classification.graph_classification import FeedForwardNN
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config
from graph4nlp.pytorch.modules.utils.generic_utils import EarlyStopping, to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger

torch.multiprocessing.set_sharing_strategy("file_system")


class TextClassifier(nn.Module):
    def __init__(self, vocab, label_model, config):
        super(TextClassifier, self).__init__()
        self.config = config
        self.vocab_model = vocab
        self.label_model = label_model
        self.graph_construction_name = self.config["model_args"]["graph_construction_name"]
        assert not (
            self.graph_construction_name in ("node_emb", "node_emb_refined")
            and config["model_args"]["graph_embedding_name"] == "gat"
        ), "dynamic graph construction does not support GAT"

        self.graph_initializer = GraphEmbeddingInitialization(
            word_vocab=self.vocab_model.in_word_vocab,
            embedding_style=config["model_args"]["graph_initialization_args"]["embedding_style"],
            hidden_size=config["model_args"]["graph_initialization_args"]["hidden_size"],
            word_dropout=config["model_args"]["graph_initialization_args"]["word_dropout"],
            rnn_dropout=config["model_args"]["graph_initialization_args"]["rnn_dropout"],
            fix_word_emb=config["model_args"]["graph_initialization_args"]["fix_word_emb"],
            fix_bert_emb=config["model_args"]["graph_initialization_args"]["fix_bert_emb"],
        )

        if self.graph_construction_name == "node_emb":
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
        elif self.graph_construction_name == "node_emb_refined":
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
                self.vocab_model.in_word_vocab.embeddings.shape[0],
                self.vocab_model.in_word_vocab.embeddings.shape[1],
                pretrained_word_emb=self.vocab_model.in_word_vocab.embeddings,
                fix_emb=config["graph_initialization_args"]["fix_word_emb"],
            ).word_emb_layer

        if config["model_args"]["graph_embedding_name"] == "gat":
            self.gnn = GAT(
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["num_layers"],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["input_size"],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "hidden_size"
                ],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "output_size"
                ],
                config["model_args"]["graph_embedding_args"]["graph_embedding_private"]["heads"],
                direction_option=config["model_args"]["graph_embedding_args"][
                    "graph_embedding_share"
                ]["direction_option"],
                feat_drop=config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "feat_drop"
                ],
                attn_drop=config["model_args"]["graph_embedding_args"]["graph_embedding_private"][
                    "attn_drop"
                ],
                negative_slope=config["model_args"]["graph_embedding_args"][
                    "graph_embedding_private"
                ]["negative_slope"],
                residual=config["model_args"]["graph_embedding_args"]["graph_embedding_private"][
                    "residual"
                ],
                activation=F.elu,
                allow_zero_in_degree=config["model_args"]["graph_embedding_args"][
                    "graph_embedding_private"
                ]["allow_zero_in_degree"],
            )
        elif config["model_args"]["graph_embedding_name"] == "graphsage":
            self.gnn = GraphSAGE(
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["num_layers"],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["input_size"],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "hidden_size"
                ],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "output_size"
                ],
                config["model_args"]["graph_embedding_args"]["graph_embedding_private"][
                    "aggregator_type"
                ],
                direction_option=config["model_args"]["graph_embedding_args"][
                    "graph_embedding_share"
                ]["direction_option"],
                feat_drop=config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "feat_drop"
                ],
                bias=config["model_args"]["graph_embedding_args"]["graph_embedding_private"][
                    "bias"
                ],
                norm=config["model_args"]["graph_embedding_args"]["graph_embedding_private"][
                    "norm"
                ],
                activation=F.relu,
                use_edge_weight=config["model_args"]["graph_embedding_args"][
                    "graph_embedding_private"
                ]["use_edge_weight"],
            )
        elif config["model_args"]["graph_embedding_name"] == "ggnn":
            self.gnn = GGNN(
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["num_layers"],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["input_size"],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "hidden_size"
                ],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "output_size"
                ],
                feat_drop=config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "feat_drop"
                ],
                direction_option=config["model_args"]["graph_embedding_args"][
                    "graph_embedding_share"
                ]["direction_option"],
                bias=config["model_args"]["graph_embedding_args"]["graph_embedding_private"][
                    "bias"
                ],
                use_edge_weight=config["model_args"]["graph_embedding_args"][
                    "graph_embedding_private"
                ]["use_edge_weight"],
            )
        elif config["model_args"]["graph_embedding_name"] == "rgcn":
            self.gnn = RGCN(
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["num_layers"],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["input_size"],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "hidden_size"
                ],
                config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "output_size"
                ],
                num_rels=config["model_args"]["graph_embedding_args"]["graph_embedding_private"][
                    "num_rels"
                ],
                direction_option=config["model_args"]["graph_embedding_args"][
                    "graph_embedding_share"
                ]["direction_option"],
                feat_drop=config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                    "feat_drop"
                ],
                regularizer="basis",
                num_bases=config["model_args"]["graph_embedding_args"]["graph_embedding_private"][
                    "num_bases"
                ],
            )
        else:
            raise RuntimeError(
                "Unknown gnn type: {}".format(config["model_args"]["graph_embedding_name"])
            )

        self.clf = FeedForwardNN(
            2 * config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["output_size"]
            if config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                "direction_option"
            ]
            == "bi_sep"
            else config["model_args"]["graph_embedding_args"]["graph_embedding_share"][
                "output_size"
            ],
            config["num_classes"],
            [config["model_args"]["graph_embedding_args"]["graph_embedding_share"]["output_size"]],
            graph_pool_type=config["model_args"]["decoder_args"]["graph_pooling_share"][
                "pooling_type"
            ],
            use_linear_proj=config["model_args"]["decoder_args"]["graph_pooling_share"][
                "max_pool_linear_proj"
            ],
        )

        self.loss = GeneralLoss("CrossEntropy")

    def forward(self, graph_list, tgt=None, require_loss=True):
        # graph embedding initialization
        batch_gd = self.graph_initializer(graph_list)

        # run dynamic graph construction if turned on
        if hasattr(self, "graph_topology") and hasattr(self.graph_topology, "dynamic_topology"):
            batch_gd = self.graph_topology.dynamic_topology(batch_gd)

        # run GNN
        self.gnn(batch_gd)

        # run graph classifier
        self.clf(batch_gd)
        logits = batch_gd.graph_attributes["logits"]

        if require_loss:
            loss = self.loss(logits, tgt)
            return logits, loss
        else:
            return logits

    def inference_forward(self, collate_data):
        return self.forward(collate_data["graph_data"], require_loss=False)

    def post_process(self, logits, label_names):
        logits_list = []

        for idx in range(len(logits)):
            logits_list.append(logits[idx].cpu().clone().numpy())

        pred_tags = [label_names[pred.argmax()] for pred in logits_list]
        return pred_tags

    @classmethod
    def load_checkpoint(cls, model_path):
        """The API to load the model.

        Parameters
        ----------
        model_path : str
            The saved model path.

        Returns
        -------
        Class
        """
        return torch.load(model_path)


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        log_config = copy.deepcopy(config)
        del log_config["env_args"]["device"]
        self.logger = Logger(
            self.config["checkpoint_args"]["out_dir"],
            config=log_config,
            overwrite=True,
        )
        self.logger.write(self.config["checkpoint_args"]["out_dir"])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        dataset = TrecDataset(
            root_dir=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["root_dir"],
            topology_subdir=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["topology_subdir"],
            graph_construction_name=self.config["model_args"]["graph_construction_name"],
            dynamic_init_graph_name=self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ].get("dynamic_init_graph_name", None),
            dynamic_init_topology_aux_args={"dummy_param": 0},
            pretrained_word_emb_name=self.config["preprocessing_args"]["pretrained_word_emb_name"],
            merge_strategy=self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ].get("merge_strategy", None),
            edge_strategy=self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ].get("edge_strategy", None),
            min_word_vocab_freq=self.config["preprocessing_args"]["min_word_freq"],
            word_emb_size=self.config["preprocessing_args"]["word_emb_size"],
            nlp_processor_args=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ].get("nlp_processor_args", None),
            seed=self.config["env_args"]["seed"],
            thread_number=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ].get("thread_number", None),
            reused_label_model=None,
        )

        self.train_dataloader = DataLoader(
            dataset.train,
            batch_size=self.config["training_args"]["batch_size"],
            shuffle=True,
            num_workers=self.config["env_args"]["num_workers"],
            collate_fn=dataset.collate_fn,
        )
        if not hasattr(dataset, "val"):
            dataset.val = dataset.test
        self.val_dataloader = DataLoader(
            dataset.val,
            batch_size=self.config["training_args"]["batch_size"],
            shuffle=False,
            num_workers=self.config["env_args"]["num_workers"],
            collate_fn=dataset.collate_fn,
        )
        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=self.config["training_args"]["batch_size"],
            shuffle=False,
            num_workers=self.config["env_args"]["num_workers"],
            collate_fn=dataset.collate_fn,
        )
        self.vocab_model = dataset.vocab_model
        self.label_model = dataset.label_model
        self.config["num_classes"] = self.label_model.num_classes
        self.num_train = len(dataset.train)
        self.num_val = len(dataset.val)
        self.num_test = len(dataset.test)
        print(
            "Train size: {}, Val size: {}, Test size: {}".format(
                self.num_train, self.num_val, self.num_test
            )
        )
        self.logger.write(
            "Train size: {}, Val size: {}, Test size: {}".format(
                self.num_train, self.num_val, self.num_test
            )
        )

    def _build_model(self):
        self.model = TextClassifier(self.vocab_model, self.label_model, self.config).to(
            self.config["env_args"]["device"]
        )

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config["training_args"]["lr"])
        self.stopper = EarlyStopping(
            os.path.join(
                self.config["checkpoint_args"]["out_dir"],
                Constants._SAVED_WEIGHTS_FILE,
            ),
            patience=self.config["training_args"]["patience"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config["training_args"]["lr_reduce_factor"],
            patience=self.config["training_args"]["lr_patience"],
            verbose=True,
        )

    def _build_evaluation(self):
        self.metric = Accuracy(["accuracy"])

    def train(self):
        dur = []
        for epoch in range(self.config["training_args"]["epochs"]):
            self.model.train()
            train_loss = []
            train_acc = []
            t0 = time.time()
            for data in self.train_dataloader:
                tgt = to_cuda(data["tgt_tensor"], self.config["env_args"]["device"])
                data["graph_data"] = data["graph_data"].to(self.config["env_args"]["device"])
                logits, loss = self.model(data["graph_data"], tgt, require_loss=True)

                # add graph regularization loss if available
                if data["graph_data"].graph_attributes.get("graph_reg", None) is not None:
                    loss = loss + data["graph_data"].graph_attributes["graph_reg"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                pred = torch.max(logits, dim=-1)[1].cpu()
                train_acc.append(
                    self.metric.calculate_scores(ground_truth=tgt.cpu(), predict=pred.cpu())[0]
                )
                dur.append(time.time() - t0)

            val_acc = self.evaluate(self.val_dataloader)
            self.scheduler.step(val_acc)
            print(
                "Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} |"
                "Train Acc: {:.4f} | Val Acc: {:.4f}".format(
                    epoch + 1,
                    self.config["training_args"]["epochs"],
                    np.mean(dur),
                    np.mean(train_loss),
                    np.mean(train_acc),
                    val_acc,
                )
            )
            self.logger.write(
                "Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} |"
                "Train Acc: {:.4f} | Val Acc: {:.4f}".format(
                    epoch + 1,
                    self.config["training_args"]["epochs"],
                    np.mean(dur),
                    np.mean(train_loss),
                    np.mean(train_acc),
                    val_acc,
                )
            )

            if self.stopper.step(val_acc, self.model):
                break

        return self.stopper.best_score

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for data in dataloader:
                tgt = to_cuda(data["tgt_tensor"], self.config["env_args"]["device"])
                data["graph_data"] = data["graph_data"].to(self.config["env_args"]["device"])
                logits = self.model(data["graph_data"], require_loss=False)
                pred_collect.append(logits)
                gt_collect.append(tgt)

            pred_collect = torch.max(torch.cat(pred_collect, 0), dim=-1)[1].cpu()
            gt_collect = torch.cat(gt_collect, 0).cpu()
            score = self.metric.calculate_scores(ground_truth=gt_collect, predict=pred_collect)[0]

            return score

    def test(self):
        # restored best saved model
        self.model = TextClassifier.load_checkpoint(self.stopper.save_model_path)

        t0 = time.time()
        acc = self.evaluate(self.test_dataloader)
        dur = time.time() - t0
        print(
            "Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f}".format(self.num_test, dur, acc)
        )
        self.logger.write(
            "Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f}".format(self.num_test, dur, acc)
        )

        return acc


def main(config):
    # configure
    np.random.seed(config["env_args"]["seed"])
    torch.manual_seed(config["env_args"]["seed"])

    if not config["env_args"]["no_cuda"] and torch.cuda.is_available():
        print("[ Using CUDA ]")
        config["env_args"]["device"] = torch.device(
            "cuda" if config["env_args"]["gpu"] < 0 else "cuda:%d" % config["env_args"]["gpu"]
        )
        torch.cuda.manual_seed(config["env_args"]["seed"])
        torch.cuda.manual_seed_all(config["env_args"]["seed"])
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        config["env_args"]["device"] = torch.device("cpu")

    print("\n" + config["checkpoint_args"]["out_dir"])

    runner = ModelHandler(config)
    t0 = time.time()

    val_acc = runner.train()
    test_acc = runner.test()

    runtime = time.time() - t0
    print("Total runtime: {:.2f}s".format(runtime))
    runner.logger.write("Total runtime: {:.2f}s\n".format(runtime))
    runner.logger.close()

    return val_acc, test_acc


################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-json_config",
        "--json_config",
        required=True,
        type=str,
        help="path to the json config file",
    )
    args = vars(parser.parse_args())

    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == "__main__":
    import multiprocessing
    import platform

    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")

    cfg = get_args()
    config = load_json_config(cfg["json_config"])
    print_config(config)

    main(config)
