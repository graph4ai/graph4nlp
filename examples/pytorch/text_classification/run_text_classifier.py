import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
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
from graph4nlp.pytorch.modules.loss.general_loss import GeneralLoss
from graph4nlp.pytorch.modules.prediction.classification.graph_classification import FeedForwardNN
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.generic_utils import EarlyStopping, grid, to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger

torch.multiprocessing.set_sharing_strategy("file_system")


class TextClassifier(nn.Module):
    def __init__(self, vocab, label_model, config):
        super(TextClassifier, self).__init__()
        self.config = config
        self.vocab_model = vocab
        self.label_model = label_model
        self.graph_name = self.config["graph_construction_args"]["graph_construction_share"][
            "graph_name"
        ]
        assert not (
            self.graph_name in ("node_emb", "node_emb_refined") and config["gnn"] == "gat"
        ), "dynamic graph construction does not support GAT"

        embedding_style = {
            "single_token_item": True if self.graph_name != "ie" else False,
            "emb_strategy": config.get("emb_strategy", "w2v_bilstm"),
            "num_rnn_layers": 1,
            "bert_model_name": config.get("bert_model_name", "bert-base-uncased"),
            "bert_lower_case": True,
        }

        self.graph_initializer = GraphEmbeddingInitialization(
            word_vocab=self.vocab_model.in_word_vocab,
            embedding_style=embedding_style,
            hidden_size=config["num_hidden"],
            word_dropout=config["word_dropout"],
            rnn_dropout=config["rnn_dropout"],
            fix_word_emb=not config["no_fix_word_emb"],
            fix_bert_emb=not config.get("no_fix_bert_emb", False),
        )

        use_edge_weight = False
        if self.graph_name == "node_emb":
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(
                sim_metric_type=config["gl_metric_type"],
                num_heads=config["gl_num_heads"],
                top_k_neigh=config["gl_top_k"],
                epsilon_neigh=config["gl_epsilon"],
                smoothness_ratio=config["gl_smoothness_ratio"],
                connectivity_ratio=config["gl_connectivity_ratio"],
                sparsity_ratio=config["gl_sparsity_ratio"],
                input_size=config["num_hidden"],
                hidden_size=config["gl_num_hidden"],
            )
            use_edge_weight = True
        elif self.graph_name == "node_emb_refined":
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
                config["init_adj_alpha"],
                sim_metric_type=config["gl_metric_type"],
                num_heads=config["gl_num_heads"],
                top_k_neigh=config["gl_top_k"],
                epsilon_neigh=config["gl_epsilon"],
                smoothness_ratio=config["gl_smoothness_ratio"],
                connectivity_ratio=config["gl_connectivity_ratio"],
                sparsity_ratio=config["gl_sparsity_ratio"],
                input_size=config["num_hidden"],
                hidden_size=config["gl_num_hidden"],
            )
            use_edge_weight = True

        if "w2v" in self.graph_initializer.embedding_layer.word_emb_layers:
            self.word_emb = self.graph_initializer.embedding_layer.word_emb_layers[
                "w2v"
            ].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                self.vocab_model.in_word_vocab.embeddings.shape[0],
                self.vocab_model.in_word_vocab.embeddings.shape[1],
                pretrained_word_emb=self.vocab_model.in_word_vocab.embeddings,
                fix_emb=not config["no_fix_word_emb"],
            ).word_emb_layer

        if config["gnn"] == "gat":
            heads = [config["gat_num_heads"]] * (config["gnn_num_layers"] - 1) + [
                config["gat_num_out_heads"]
            ]
            self.gnn = GAT(
                config["gnn_num_layers"],
                config["num_hidden"],
                config["num_hidden"],
                config["num_hidden"],
                heads,
                direction_option=config["gnn_direction_option"],
                feat_drop=config["gnn_dropout"],
                attn_drop=config["gat_attn_dropout"],
                negative_slope=config["gat_negative_slope"],
                residual=config["gat_residual"],
                activation=F.elu,
                allow_zero_in_degree=True,
            )
        elif config["gnn"] == "graphsage":
            self.gnn = GraphSAGE(
                config["gnn_num_layers"],
                config["num_hidden"],
                config["num_hidden"],
                config["num_hidden"],
                config["graphsage_aggreagte_type"],
                direction_option=config["gnn_direction_option"],
                feat_drop=config["gnn_dropout"],
                bias=True,
                norm=None,
                activation=F.relu,
                use_edge_weight=use_edge_weight,
            )
        elif config["gnn"] == "ggnn":
            self.gnn = GGNN(
                config["gnn_num_layers"],
                config["num_hidden"],
                config["num_hidden"],
                config["num_hidden"],
                feat_drop=config["gnn_dropout"],
                direction_option=config["gnn_direction_option"],
                bias=True,
                use_edge_weight=use_edge_weight,
            )
        else:
            raise RuntimeError("Unknown gnn type: {}".format(config["gnn"]))

        self.clf = FeedForwardNN(
            2 * config["num_hidden"]
            if config["gnn_direction_option"] == "bi_sep"
            else config["num_hidden"],
            config["num_classes"],
            [config["num_hidden"]],
            graph_pool_type=config["graph_pooling"],
            dim=config["num_hidden"],
            use_linear_proj=config["max_pool_linear_proj"],
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
        self.logger = Logger(
            self.config["out_dir"],
            config={k: v for k, v in self.config.items() if k != "device"},
            overwrite=True,
        )
        self.logger.write(self.config["out_dir"])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        self.graph_name = self.config["graph_construction_args"]["graph_construction_share"][
            "graph_name"
        ]
        topology_subdir = "{}_graph".format(self.graph_name)
        if self.graph_name == "node_emb_refined":
            topology_subdir += "_{}".format(
                self.config["graph_construction_args"]["graph_construction_private"][
                    "dynamic_init_graph_name"
                ]
            )

        dataset = TrecDataset(
            root_dir=self.config["graph_construction_args"]["graph_construction_share"]["root_dir"],
            topology_subdir=topology_subdir,
            graph_name=self.graph_name,
            dynamic_init_graph_name=self.config["graph_construction_args"][
                "graph_construction_private"
            ]["dynamic_init_graph_name"],
            dynamic_init_topology_aux_args={"dummy_param": 0},
            pretrained_word_emb_name=self.config["pretrained_word_emb_name"],
            merge_strategy=self.config["graph_construction_args"]["graph_construction_private"][
                "merge_strategy"
            ],
            edge_strategy=self.config["graph_construction_args"]["graph_construction_private"][
                "edge_strategy"
            ],
            min_word_vocab_freq=self.config.get("min_word_freq", 1),
            word_emb_size=self.config.get("word_emb_size", 300),
            seed=self.config["seed"],
            thread_number=self.config["graph_construction_args"]["graph_construction_share"][
                "thread_number"
            ],
            port=self.config["graph_construction_args"]["graph_construction_share"]["port"],
            timeout=self.config["graph_construction_args"]["graph_construction_share"]["timeout"],
            reused_label_model=None,
        )

        self.train_dataloader = DataLoader(
            dataset.train,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            collate_fn=dataset.collate_fn,
        )
        if not hasattr(dataset, "val"):
            dataset.val = dataset.test
        self.val_dataloader = DataLoader(
            dataset.val,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=dataset.collate_fn,
        )
        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
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
            self.config["device"]
        )

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config["lr"])
        self.stopper = EarlyStopping(
            os.path.join(
                self.config["out_dir"],
                self.config.get("model_ckpt_name", Constants._SAVED_WEIGHTS_FILE),
            ),
            patience=self.config["patience"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config["lr_reduce_factor"],
            patience=self.config["lr_patience"],
            verbose=True,
        )

    def _build_evaluation(self):
        self.metric = Accuracy(["accuracy"])

    def train(self):
        dur = []
        for epoch in range(self.config["epochs"]):
            self.model.train()
            train_loss = []
            train_acc = []
            t0 = time.time()
            for data in self.train_dataloader:
                tgt = to_cuda(data["tgt_tensor"], self.config["device"])
                data["graph_data"] = data["graph_data"].to(self.config["device"])
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
                    self.config["epochs"],
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
                    self.config["epochs"],
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
                tgt = to_cuda(data["tgt_tensor"], self.config["device"])
                data["graph_data"] = data["graph_data"].to(self.config["device"])
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
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if not config["no_cuda"] and torch.cuda.is_available():
        print("[ Using CUDA ]")
        config["device"] = torch.device("cuda" if config["gpu"] < 0 else "cuda:%d" % config["gpu"])
        torch.cuda.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        config["device"] = torch.device("cpu")

    print("\n" + config["out_dir"])

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
        "-config", "--config", required=True, type=str, help="path to the config file"
    )
    parser.add_argument("--grid_search", action="store_true", help="flag: grid search")
    args = vars(parser.parse_args())

    return args


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.safe_load(setting)

    return config


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def grid_search_main(config):
    grid_search_hyperparams = []
    log_path = config["out_dir"]
    for k, v in config.items():
        if isinstance(v, list):
            grid_search_hyperparams.append(k)
            log_path += "_{}_{}".format(k, v)

    logger = Logger(log_path, config=config, overwrite=True)

    best_config = None
    best_score = -1
    configs = grid(config)
    for cnf in configs:
        for k in grid_search_hyperparams:
            cnf["out_dir"] += "_{}_{}".format(k, cnf[k])

        val_score, test_score = main(cnf)
        if best_score < test_score:
            best_score = test_score
            best_config = cnf
            print("Found a better configuration: {}".format(best_score))
            logger.write("Found a better configuration: {}".format(best_score))

    print("\nBest configuration:")
    logger.write("\nBest configuration:")
    for k in grid_search_hyperparams:
        print("{}: {}".format(k, best_config[k]))
        logger.write("{}: {}".format(k, best_config[k]))

    print("Best score: {}".format(best_score))
    logger.write("Best score: {}\n".format(best_score))
    logger.close()


if __name__ == "__main__":
    import platform
    import multiprocessing

    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")

    cfg = get_args()
    config = get_config(cfg["config"])
    print_config(config)
    if cfg["grid_search"]:
        grid_search_main(config)
    else:
        main(config)
