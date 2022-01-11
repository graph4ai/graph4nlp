import argparse
import multiprocessing
import os
import platform
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.squad import SQuADDataset
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.models.graph2seq_loss import Graph2SeqLoss
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.evaluation import BLEU, METEOR, ROUGE
from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import (
    WordEmbedding,
)
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import get_yaml_config, update_values
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.utils.generic_utils import EarlyStopping, grid, to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger

from .fused_embedding_construction import FusedEmbeddingConstruction


class QGModel(nn.Module):
    def __init__(self, vocab, config):
        super(QGModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.use_coverage = self.config["decoder_args"]["rnn_decoder_share"]["use_coverage"]

        # build Graph2Seq model
        self.g2s = Graph2Seq.from_args(config, self.vocab)

        if "w2v" in self.g2s.graph_initializer.embedding_layer.word_emb_layers:
            self.word_emb = self.g2s.graph_initializer.embedding_layer.word_emb_layers[
                "w2v"
            ].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                self.vocab.in_word_vocab.embeddings.shape[0],
                self.vocab.in_word_vocab.embeddings.shape[1],
                pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                fix_emb=config["graph_initialization_args"]["fix_word_emb"],
            ).word_emb_layer
        self.g2s.seq_decoder.tgt_emb = self.word_emb

        self.loss_calc = Graph2SeqLoss(
            ignore_index=self.vocab.out_word_vocab.PAD,
            use_coverage=self.use_coverage,
            coverage_weight=config["coverage_loss_ratio"],
        )

        # Replace the default embedding construction layer
        #   with the customized passage-answer alignment embedding construction layer
        # TODO: delete the default layer and clear the memory
        embedding_styles = config["graph_initialization_args"]["embedding_style"]
        self.g2s.graph_initializer.embedding_layer = FusedEmbeddingConstruction(
            self.vocab.in_word_vocab,
            embedding_styles["single_token_item"],
            emb_strategy=embedding_styles["emb_strategy"],
            hidden_size=config["graph_initialization_args"]["hidden_size"],
            num_rnn_layers=embedding_styles.get("num_rnn_layers", 1),
            fix_word_emb=config["graph_initialization_args"]["fix_word_emb"],
            fix_bert_emb=config["graph_initialization_args"]["fix_bert_emb"],
            bert_model_name=embedding_styles.get("bert_model_name", "bert-base-uncased"),
            bert_lower_case=embedding_styles.get("bert_lower_case", True),
            word_dropout=config["graph_initialization_args"]["word_dropout"],
            bert_dropout=config["graph_initialization_args"].get("bert_dropout", None),
            rnn_dropout=config["graph_initialization_args"]["rnn_dropout"],
        )
        self.graph_name = self.g2s.graph_name
        self.vocab_model = self.g2s.vocab_model

    def encode_init_node_feature(self, data):
        # graph embedding initialization
        batch_gd = self.g2s.graph_initializer(data)
        return batch_gd

    def forward(self, data, oov_dict=None, require_loss=True):
        batch_gd = self.encode_init_node_feature(data)
        if require_loss:
            tgt = data["tgt_tensor"]
        else:
            tgt = None
        prob, enc_attn_weights, coverage_vectors = self.g2s.encoder_decoder(
            batch_gd, oov_dict=oov_dict, tgt_seq=tgt
        )

        if require_loss:
            tgt = data["tgt_tensor"]
            min_length = min(prob.shape[1], tgt.shape[1])
            prob = prob[:, :min_length, :]
            tgt = tgt[:, :min_length]
            loss = self.loss_calc(
                prob,
                label=tgt,
                enc_attn_weights=enc_attn_weights,
                coverage_vectors=coverage_vectors,
            )
            return prob, loss * min_length / 2
        else:
            return prob

    def inference_forward(self, data, beam_size, topk=1, oov_dict=None):
        batch_gd = self.encode_init_node_feature(data)
        return self.g2s.encoder_decoder_beam_search(
            batch_graph=batch_gd, beam_size=beam_size, topk=topk, oov_dict=oov_dict
        )

    def post_process(self, decode_results, vocab):
        return self.g2s.post_process(decode_results, vocab)


class ModelHandler:
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self.use_copy = self.config["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_coverage = self.config["decoder_args"]["rnn_decoder_share"]["use_coverage"]
        self.logger = Logger(
            config["out_dir"],
            config={k: v for k, v in config.items() if k != "device"},
            overwrite=True,
        )
        self.logger.write(config["out_dir"])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        dataset = SQuADDataset(
            root_dir=self.config["graph_construction_args"]["graph_construction_share"]["root_dir"],
            topology_subdir=self.config["graph_construction_args"]["graph_construction_share"][
                "topology_subdir"
            ],
            graph_name=self.config["graph_construction_args"]["graph_construction_share"][
                "graph_name"
            ],
            dynamic_init_graph_name=self.config["graph_construction_args"][
                "graph_construction_private"
            ].get("dynamic_init_graph_type", None),
            dynamic_init_topology_aux_args={"dummy_param": 0},
            pretrained_word_emb_name=self.config["pretrained_word_emb_name"],
            merge_strategy=self.config["graph_construction_args"]["graph_construction_private"][
                "merge_strategy"
            ],
            edge_strategy=self.config["graph_construction_args"]["graph_construction_private"][
                "edge_strategy"
            ],
            max_word_vocab_size=self.config["top_word_vocab"],
            min_word_vocab_freq=self.config["min_word_freq"],
            word_emb_size=self.config["word_emb_size"],
            share_vocab=self.config["share_vocab"],
            seed=self.config["seed"],
            thread_number=self.config["graph_construction_args"]["graph_construction_share"][
                "thread_number"
            ],
            port=self.config["graph_construction_args"]["graph_construction_share"]["port"],
            timeout=self.config["graph_construction_args"]["graph_construction_share"]["timeout"],
        )

        self.train_dataloader = DataLoader(
            dataset.train,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            collate_fn=dataset.collate_fn,
        )
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
        self.vocab = dataset.vocab_model
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
        self.model = QGModel(self.vocab, self.config).to(self.config["device"])

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config["lr"])
        self.stopper = EarlyStopping(
            os.path.join(self.config["out_dir"], Constants._SAVED_WEIGHTS_FILE),
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
        self.metrics = {"BLEU": BLEU(n_grams=[1, 2, 3, 4]), "METEOR": METEOR(), "ROUGE": ROUGE()}

    def train(self):
        for epoch in range(self.config["epochs"]):
            self.model.train()
            train_loss = []
            dur = []
            t0 = time.time()
            for i, data in enumerate(self.train_dataloader):
                data = all_to_cuda(data, self.config["device"])
                data["graph_data"] = data["graph_data"].to(self.config["device"])
                oov_dict = None
                if self.use_copy:
                    oov_dict, tgt = prepare_ext_vocab(
                        data["graph_data"],
                        self.vocab,
                        gt_str=data["tgt_text"],
                        device=self.config["device"],
                    )
                    data["tgt_tensor"] = tgt

                logits, loss = self.model(data, oov_dict=oov_dict, require_loss=True)
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.get("grad_clipping", None) not in (None, 0):
                    # Clip gradients
                    parameters = [p for p in self.model.parameters() if p.requires_grad]

                    torch.nn.utils.clip_grad_norm_(parameters, self.config["grad_clipping"])

                self.optimizer.step()
                train_loss.append(loss.item())

                # pred = torch.max(logits, dim=-1)[1].cpu()
                dur.append(time.time() - t0)
                if (i + 1) % 100 == 0:
                    format_str = (
                        "Epoch: [{} / {}] | Step: {} / {} | Time: {:.2f}s | Loss: {:.4f} |"
                        " Val scores:".format(
                            epoch + 1,
                            self.config["epochs"],
                            i,
                            len(self.train_dataloader),
                            np.mean(dur),
                            np.mean(train_loss),
                        )
                    )
                    print(format_str)
                    self.logger.write(format_str)

            val_scores = self.evaluate(self.val_dataloader)
            if epoch > 15:
                self.scheduler.step(val_scores[self.config["early_stop_metric"]])
            format_str = "Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} | Val scores:".format(
                epoch + 1, self.config["epochs"], np.mean(dur), np.mean(train_loss)
            )
            format_str += self.metric_to_str(val_scores)
            print(format_str)
            self.logger.write(format_str)

            if epoch > 0 and self.stopper.step(
                val_scores[self.config["early_stop_metric"]], self.model
            ):
                break

        return self.stopper.best_score

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for data in dataloader:
                data = all_to_cuda(data, self.config["device"])
                data["graph_data"] = data["graph_data"].to(self.config["device"])

                if self.use_copy:
                    oov_dict = prepare_ext_vocab(
                        data["graph_data"], self.vocab, device=self.config["device"]
                    )
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                prob = self.model(data, oov_dict=oov_dict, require_loss=False)
                pred = prob.argmax(dim=-1)

                pred_str = wordid2str(pred.detach().cpu(), ref_dict)
                pred_collect.extend(pred_str)
                gt_collect.extend(data["tgt_text"])

            scores = self.evaluate_predictions(gt_collect, pred_collect)
            return scores

    def translate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for data in dataloader:
                data = all_to_cuda(data, self.config["device"])
                data["graph_data"] = data["graph_data"].to(self.config["device"])
                if self.use_copy:
                    oov_dict = prepare_ext_vocab(
                        data["graph_data"], self.vocab, device=self.config["device"]
                    )
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                batch_gd = self.model.encode_init_node_feature(data)
                prob = self.model.g2s.encoder_decoder_beam_search(
                    batch_gd, self.config["beam_size"], topk=1, oov_dict=oov_dict
                )

                pred_ids = (
                    torch.zeros(
                        len(prob),
                        self.config["decoder_args"]["rnn_decoder_private"]["max_decoder_step"],
                    )
                    .fill_(ref_dict.EOS)
                    .to(self.config["device"])
                    .int()
                )
                for i, item in enumerate(prob):
                    item = item[0]
                    seq = [j.view(1, 1) for j in item]
                    seq = torch.cat(seq, dim=1)
                    pred_ids[i, : seq.shape[1]] = seq

                pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)

                pred_collect.extend(pred_str)
                gt_collect.extend(data["tgt_text"])

            scores = self.evaluate_predictions(gt_collect, pred_collect)

            return scores

    def test(self):
        # restored best saved model
        self.model = torch.load(
            os.path.join(self.config["out_dir"], Constants._SAVED_WEIGHTS_FILE)
        ).to(self.config["device"])

        t0 = time.time()
        scores = self.translate(self.test_dataloader)
        dur = time.time() - t0
        format_str = "Test examples: {} | Time: {:.2f}s |  Test scores:".format(self.num_test, dur)
        format_str += self.metric_to_str(scores)
        print(format_str)
        self.logger.write(format_str)

        return scores

    def evaluate_predictions(self, ground_truth, predict):
        output = {}
        for name, scorer in self.metrics.items():
            score = scorer.calculate_scores(ground_truth=ground_truth, predict=predict)
            if name.upper() == "BLEU":
                for i in range(len(score[0])):
                    output["BLEU_{}".format(i + 1)] = score[0][i]
            else:
                output[name] = score[0]

        return output

    def metric_to_str(self, metrics):
        format_str = ""
        for k in metrics:
            format_str += " {} = {:0.5f},".format(k.upper(), metrics[k])

        return format_str[:-1]


def main(config):
    # configure
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if not config["no_cuda"] and torch.cuda.is_available():
        print("[ Using CUDA ]")
        config["device"] = torch.device("cuda" if config["gpu"] < 0 else "cuda:%d" % config["gpu"])
        cudnn.benchmark = True
        torch.cuda.manual_seed(config["seed"])
    else:
        config["device"] = torch.device("cpu")

    print("\n" + config["out_dir"])

    runner = ModelHandler(config)
    t0 = time.time()

    val_score = runner.train()
    test_scores = runner.test()

    # print('Removed best saved model file to save disk space')
    # os.remove(runner.stopper.save_model_path)
    runtime = time.time() - t0
    print("Total runtime: {:.2f}s".format(time.time() - t0))
    runner.logger.write("Total runtime: {:.2f}s\n".format(runtime))
    runner.logger.close()

    return val_score, test_scores


def wordid2str(word_ids, vocab):
    ret = []
    assert len(word_ids.shape) == 2, print(word_ids.shape)
    for i in range(word_ids.shape[0]):
        id_list = word_ids[i, :]
        ret_inst = []
        for j in range(id_list.shape[0]):
            if id_list[j] == vocab.EOS or id_list[j] == vocab.PAD:
                break
            token = vocab.getWord(id_list[j])
            ret_inst.append(token)
        ret.append(" ".join(ret_inst))
    return ret


def all_to_cuda(data, device=None):
    if isinstance(data, torch.Tensor):
        data = to_cuda(data, device)
    elif isinstance(data, (list, dict)):
        keys = range(len(data)) if isinstance(data, list) else data.keys()
        for k in keys:
            if isinstance(data[k], torch.Tensor):
                data[k] = to_cuda(data[k], device)

    return data


################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-task_config", "--task_config", required=True, type=str, help="path to the config file"
    )
    parser.add_argument(
        "-g2s_config", "--g2s_config", required=True, type=str, help="path to the config file"
    )
    parser.add_argument("--grid_search", action="store_true", help="flag: grid search")
    args = vars(parser.parse_args())

    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->  {}".format(keystr, val))
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
    best_scores = None
    configs = grid(config)
    for cnf in configs:
        for k in grid_search_hyperparams:
            cnf["out_dir"] += "_{}_{}".format(k, cnf[k])

        val_score, test_scores = main(cnf)
        if best_score < test_scores[cnf["early_stop_metric"]]:
            best_score = test_scores[cnf["early_stop_metric"]]
            best_scores = test_scores
            best_config = cnf
            print("Found a better configuration: {}".format(best_scores))
            logger.write("Found a better configuration: {}".format(best_scores))

    print("\nBest configuration:")
    logger.write("\nBest configuration:")
    for k in grid_search_hyperparams:
        print("{}: {}".format(k, best_config[k]))
        logger.write("{}: {}".format(k, best_config[k]))

    print("Best score: {}".format(best_scores))
    logger.write("Best score: {}\n".format(best_scores))
    logger.close()


if __name__ == "__main__":

    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")

    cfg = get_args()
    task_args = get_yaml_config(cfg["task_config"])
    g2s_args = get_yaml_config(cfg["g2s_config"])
    # load Graph2Seq template config
    g2s_template = get_basic_args(
        graph_construction_name=g2s_args["graph_construction_name"],
        graph_embedding_name=g2s_args["graph_embedding_name"],
        decoder_name=g2s_args["decoder_name"],
    )
    update_values(to_args=g2s_template, from_args_list=[g2s_args, task_args])
    print_config(g2s_template)
    if cfg["grid_search"]:
        grid_search_main(g2s_template)
    else:
        main(g2s_template)
