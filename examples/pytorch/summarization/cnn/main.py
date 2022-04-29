import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.cnn import CNNDataset
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.models.graph2seq_loss import Graph2SeqLoss
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.evaluation.rouge import ROUGE
from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import WordEmbedding
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import get_yaml_config, update_values
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.utils.generic_utils import EarlyStopping, to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.modules.utils.summarization_utils import wordid2str


def all_to_cuda(data, device=None):
    if isinstance(data, torch.Tensor):
        data = to_cuda(data, device)
    elif isinstance(data, (list, dict)):
        keys = range(len(data)) if isinstance(data, list) else data.keys()
        for k in keys:
            if isinstance(data[k], torch.Tensor):
                data[k] = to_cuda(data[k], device)

    return data


class SumModel(nn.Module):
    def __init__(self, vocab, config):
        super(SumModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.use_coverage = self.config["decoder_args"]["rnn_decoder_share"]["use_coverage"]

        # build Graph2Seq model
        self.g2s = Graph2Seq.from_args(config, self.vocab)

        self.graph_name = self.config["graph_construction_args"]["graph_construction_share"][
            "graph_name"
        ]

        if "w2v" in self.g2s.graph_initializer.embedding_layer.word_emb_layers:
            self.word_emb = self.g2s.graph_initializer.embedding_layer.word_emb_layers[
                "w2v"
            ].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                self.vocab.in_word_vocab.embeddings.shape[0],
                self.vocab.in_word_vocab.embeddings.shape[1],
                pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                fix_emb=config["graph_construction_args"]["node_embedding"]["fix_word_emb"],
            ).word_emb_layer

        self.g2s.seq_decoder.tgt_emb = self.word_emb

        self.loss_calc = Graph2SeqLoss(
            ignore_index=self.vocab.out_word_vocab.PAD,
            use_coverage=self.use_coverage,
            coverage_weight=config["coverage_loss_ratio"],
        )

    def forward(self, data, oov_dict=None, require_loss=True):
        if require_loss:
            prob, enc_attn_weights, coverage_vectors = self.g2s(
                data["graph_data"], oov_dict=oov_dict, tgt_seq=data["tgt_seq"]
            )

            tgt = data["tgt_seq"]
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
            prob, enc_attn_weights, coverage_vectors = self.g2s(
                data["graph_data"], oov_dict=oov_dict
            )
            return prob

    def inference_forward(self, batch_graph, beam_size, topk, oov_dict):
        return self.g2s.inference_forward(
            batch_graph, beam_size=beam_size, topk=topk, oov_dict=oov_dict
        )

    def post_process(self, decode_results, vocab):
        return self.g2s.post_process(decode_results=decode_results, vocab=vocab)

    @classmethod
    def load_checkpoint(cls, model_path):
        return torch.load(model_path)


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
        dataset = CNNDataset(
            root_dir=self.config["graph_construction_args"]["graph_construction_share"]["root_dir"],
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
            lower_case=self.config["vocab_lower_case"],
            seed=self.config["seed"],
            topology_subdir=self.config["graph_construction_args"]["graph_construction_share"][
                "topology_subdir"
            ],
            graph_name=self.config["graph_construction_args"]["graph_construction_share"][
                "graph_name"
            ],
            thread_number=self.config["graph_construction_args"]["graph_construction_share"][
                "thread_number"
            ],
            port=self.config["graph_construction_args"]["graph_construction_share"]["port"],
            timeout=self.config["graph_construction_args"]["graph_construction_share"]["timeout"],
            tokenizer=None,
        )

        dataset.train = dataset.train[: self.config["n_samples"]]
        dataset.val = dataset.val[: self.config["n_samples"]]
        dataset.test = dataset.test[: self.config["n_samples"]]

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
        # self.model = Graph2Seq.from_args(self.config, self.vocab, self.config['device'])
        self.model = SumModel(self.vocab, self.config).to(self.config["device"])

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
        self.metrics = {"ROUGE": ROUGE()}

    def train(self):
        dur = []
        for epoch in range(self.config["epochs"]):
            self.model.train()
            train_loss = []
            t0 = time.time()

            for i, data in enumerate(self.train_dataloader):
                data = all_to_cuda(data, self.config["device"])
                to_cuda(data["graph_data"], self.config["device"])

                oov_dict = None
                if self.use_copy:
                    oov_dict, tgt = prepare_ext_vocab(
                        data["graph_data"],
                        self.vocab,
                        gt_str=data["output_str"],
                        device=self.config["device"],
                    )
                    data["tgt_seq"] = tgt

                logits, loss = self.model(data, oov_dict=oov_dict, require_loss=True)
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.get("grad_clipping", None) not in (None, 0):
                    # Clip gradients
                    parameters = [p for p in self.model.parameters() if p.requires_grad]

                    torch.nn.utils.clip_grad_norm_(parameters, self.config["grad_clipping"])

                self.optimizer.step()
                train_loss.append(loss.item())
                print("Epoch = {}, Step = {}, Loss = {:.3f}".format(epoch, i, loss.item()))

                dur.append(time.time() - t0)

            val_scores = self.evaluate(self.val_dataloader)
            self.scheduler.step(val_scores[self.config["early_stop_metric"]])
            format_str = "Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} | Val scores:".format(
                epoch + 1, self.config["epochs"], np.mean(dur), np.mean(train_loss)
            )
            format_str += self.metric_to_str(val_scores)
            print(format_str)
            self.logger.write(format_str)

            if self.stopper.step(val_scores[self.config["early_stop_metric"]], self.model):
                break

        return self.stopper.best_score

    def evaluate(self, dataloader, write2file=False, part="dev"):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for data in dataloader:
                data = all_to_cuda(data, self.config["device"])
                to_cuda(data["graph_data"], self.config["device"])

                if self.use_copy:
                    oov_dict, tgt = prepare_ext_vocab(
                        data["graph_data"],
                        self.vocab,
                        gt_str=data["output_str"],
                        device=self.config["device"],
                    )
                    data["tgt_text"] = tgt
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                prob = self.model(data, oov_dict=oov_dict, require_loss=False)
                pred = prob.argmax(dim=-1)

                pred_str = wordid2str(pred.detach().cpu(), ref_dict)
                pred_collect.extend(pred_str)
                gt_collect.extend(data["output_str"])

            if write2file:
                with open(
                    "{}/{}_pred.txt".format(
                        self.config["out_dir"], self.config["out_dir"].split("/")[-1]
                    ),
                    "w+",
                ) as f:
                    for line in pred_collect:
                        f.write(line + "\n")

                with open(
                    "{}/{}_gt.txt".format(
                        self.config["out_dir"], self.config["out_dir"].split("/")[-1]
                    ),
                    "w+",
                ) as f:
                    for line in gt_collect:
                        f.write(line + "\n")

            scores = self.evaluate_predictions(gt_collect, pred_collect)

            return scores

    def translate(self, dataloader, write2file=True):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for i, data in enumerate(dataloader):
                print(i)
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
                batch_graph = self.model.g2s.graph_topology(data["graph_data"])
                prob = self.model.g2s.encoder_decoder_beam_search(
                    batch_graph, self.config["beam_size"], topk=1, oov_dict=oov_dict
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
                gt_collect.extend(data["output_str"])

            if write2file:
                with open(
                    "{}/{}_bs{}_pred.txt".format(
                        self.config["out_dir"],
                        self.config["out_dir"].split("/")[-1],
                        self.config["beam_size"],
                    ),
                    "w+",
                ) as f:
                    for line in pred_collect:
                        f.write(line + "\n")

                with open(
                    "{}/{}_bs{}_gt.txt".format(
                        self.config["out_dir"],
                        self.config["out_dir"].split("/")[-1],
                        self.config["beam_size"],
                    ),
                    "w+",
                ) as f:
                    for line in gt_collect:
                        f.write(line + "\n")

            scores = self.evaluate_predictions(gt_collect, pred_collect)

            return scores

    def test(self):
        # restored best saved model
        self.model.load_checkpoint(self.stopper.save_model_path)

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
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


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
    # greedy search
    # runner.stopper.load_checkpoint(runner.model)
    # test_scores = runner.evaluate(runner.test_dataloader, write2file=True, part='test')
    # beam search
    test_scores = runner.test()

    # print('Removed best saved model file to save disk space')
    # os.remove(runner.stopper.save_model_path)
    runtime = time.time() - t0
    print("Total runtime: {:.2f}s".format(time.time() - t0))
    runner.logger.write("Total runtime: {:.2f}s\n".format(runtime))
    runner.logger.close()

    return val_score, test_scores


if __name__ == "__main__":
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
    main(g2s_template)
