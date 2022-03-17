import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing

from graph4nlp.pytorch.data.dataset import SequenceLabelingDataItem
from graph4nlp.pytorch.inference_wrapper.classifier_inference_wrapper import (
    ClassifierInferenceWrapper,
)
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.utils.config_utils import get_yaml_config, update_values
from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda

from conll import ConllDataset_inference
from conlleval import evaluate
from line_graph_construction import LineBasedGraphConstruction
from model import Word2tag

torch.multiprocessing.set_sharing_strategy("file_system")
cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def all_to_cuda(data, device=None):
    if isinstance(data, torch.Tensor):
        data = to_cuda(data, device)
    elif isinstance(data, (list, dict)):
        keys = range(len(data)) if isinstance(data, list) else data.keys()
        for k in keys:
            if isinstance(data[k], torch.Tensor):
                data[k] = to_cuda(data[k], device)

    return data


def conll_score(preds, tgts, tag_types):
    # preds is a list and each elements is the list of tags of a sentence
    # tgts is a lits and each elements is the tensor of tags of a text
    pred_list = []
    tgt_list = []

    for idx in range(len(preds)):
        pred_list.append(preds[idx].cpu().clone().numpy())
    for idx in range(len(tgts)):
        tgt_list.extend(tgts[idx].cpu().clone().numpy().tolist())
    pred_tags = [tag_types[int(pred)] for pred in pred_list]
    tgt_tags = [tag_types[int(tgt)] for tgt in tgt_list]
    prec, rec, f1 = evaluate(tgt_tags, pred_tags, verbose=False)
    return prec, rec, f1


def logits2tag(logits):
    _, pred = torch.max(logits, dim=-1)
    # print(pred.size())
    return pred


def write_file(tokens_collect, pred_collect, tag_collect, file_name, tag_types):
    num_sent = len(tokens_collect)
    f = open(file_name, "w")
    for idx in range(num_sent):
        sent_token = tokens_collect[idx]
        sent_pred = pred_collect[idx].cpu().clone().numpy()
        sent_tag = tag_collect[idx].cpu().clone().numpy()
        # f.write('%s\n' % ('-X- SENTENCE START'))
        for word_idx in range(len(sent_token)):
            w = sent_token[word_idx]
            tgt = tag_types[sent_tag[word_idx].item()]
            pred = tag_types[sent_pred[word_idx].item()]
            f.write("%d %s %s %s\n" % (word_idx + 1, w, tgt, pred))

    f.close()


def get_tokens(g_list):
    tokens = []
    for g in g_list:
        sent_token = []
        dic = g.node_attributes
        for node in dic:
            sent_token.append(node["token"])
            if "ROOT" in sent_token:
                sent_token.remove("ROOT")
        tokens.append(sent_token)
    return tokens


class Conll:
    def __init__(self, opt):
        super(Conll, self).__init__()
        self.tag_types = ["I-PER", "O", "B-ORG", "B-LOC", "I-ORG", "I-MISC", "I-LOC", "B-MISC"]
        if args.gpu > -1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.checkpoint_save_path = "./checkpoints/"

        print("finish building model")
        self.opt = opt
        self._build_model()

    def _build_model(self):
        self.model = Word2tag.load_checkpoint(self.checkpoint_save_path, "best.pt").to(self.device)

        self.inference_tool = ClassifierInferenceWrapper(
            cfg=self.opt,
            model=self.model,
            label_names=self.tag_types,
            dataset=ConllDataset_inference,
            data_item=SequenceLabelingDataItem,
            topology_builder=LineBasedGraphConstruction,
            lower_case=True,
            tokenizer=None,
        )

    def predict(self):
        """The input of the self.inference_tool.predict() is a list of sentence.
        The default tokenizer is based on white space (if tokenizer is None).
        The user can also customize their own tokenizer in defining the \
            ClassifierInferenceWrapper."""
        self.model.eval()
        ret = self.inference_tool.predict(
            raw_contents=["there is a list of jobs", "good morning"], batch_size=1
        )
        print(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    parser.add_argument(
        "--direction_option",
        type=str,
        default="bi_fuse",
        help="direction type (`undirected`, `bi_fuse`, `bi_sep`)",
    )
    parser.add_argument(
        "--lstm_num_layers", type=int, default=1, help="number of hidden layers in lstm"
    )
    parser.add_argument(
        "--gnn_num_layers", type=int, default=1, help="number of hidden layers in gnn"
    )
    parser.add_argument("--init_hidden_size", type=int, default=300, help="initial_emb_hidden_size")
    parser.add_argument("--hidden_size", type=int, default=128, help="initial_emb_hidden_size")
    parser.add_argument("--lstm_hidden_size", type=int, default=80, help="initial_emb_hidden_size")
    parser.add_argument("--num_class", type=int, default=8, help="num_class")
    parser.add_argument(
        "--residual", action="store_true", default=False, help="use residual connection"
    )
    parser.add_argument("--word_dropout", type=float, default=0.5, help="input feature dropout")
    parser.add_argument("--tag_dropout", type=float, default=0.5, help="input feature dropout")
    parser.add_argument(
        "--rnn_dropout", type=list, default=0.33, help="dropout for rnn in word_emb"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-5, help="weight decay")
    parser.add_argument(
        "--aggregate_type",
        type=str,
        default="mean",
        help="aggregate type: 'mean','gcn','pool','lstm'",
    )
    parser.add_argument(
        "--gnn_type", type=str, default="graphsage", help="ingnn type: 'gat','graphsage','ggnn'"
    )
    parser.add_argument("--use_gnn", type=bool, default=True, help="whether to use gnn")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size for training")
    parser.add_argument(
        "--graph_name",
        type=str,
        default="line_graph",
        help="graph_type:line_graph, dependency_graph, dynamic_graph",
    )
    parser.add_argument(
        "--static_or_dynamic",
        type=str,
        default="static",
        help="static or dynamic",
    )
    parser.add_argument(
        "--init_graph_type",
        type=str,
        default="line",
        help="initial graph construction type ('line', 'dependency', 'constituency', 'ie')",
    )
    parser.add_argument(
        "--pre_word_emb_file", type=str, default=None, help="path of pretrained_word_emb_file"
    )
    parser.add_argument(
        "--gl_num_heads", type=int, default=1, help="num of heads for dynamic graph construction"
    )
    parser.add_argument(
        "--gl_epsilon", type=int, default=0.5, help="epsilon for graph sparsification"
    )
    parser.add_argument("--gl_top_k", type=int, default=None, help="top k for graph sparsification")
    parser.add_argument(
        "--gl_smoothness_ratio",
        type=float,
        default=None,
        help="smoothness ratio for graph regularization loss",
    )
    parser.add_argument(
        "--gl_sparsity_ratio",
        type=float,
        default=None,
        help="sparsity ratio for graph regularization loss",
    )
    parser.add_argument(
        "--gl_connectivity_ratio",
        type=float,
        default=None,
        help="connectivity ratio for graph regularization loss",
    )
    parser.add_argument(
        "--init_adj_alpha",
        type=float,
        default=0.8,
        help="alpha ratio for combining initial graph adjacency matrix",
    )
    parser.add_argument(
        "--gl_metric_type",
        type=str,
        default="weighted_cosine",
        help="similarity metric type for dynamic graph construction ('weighted_cosine', 'attention', \
            'rbf_kernel', 'cosine')",
    )
    parser.add_argument(
        "--no_fix_word_emb",
        type=bool,
        default=False,
        help="Not fix pretrained word embeddings (default: false)",
    )
    parser.add_argument(
        "--no_fix_bert_emb",
        type=bool,
        default=False,
        help="Not fix pretrained word embeddings (default: false)",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="./ner_inference.yaml",
        help="Not fix pretrained word embeddings (default: false)",
    )

    import datetime

    starttime = datetime.datetime.now()
    # long running
    # do something other

    args = parser.parse_args()
    print("load ner template config")
    ner_args = get_yaml_config(args.task_config)

    ner_template = get_basic_args(
        graph_construction_name=ner_args["graph_construction_name"],
        graph_embedding_name=ner_args["graph_embedding_name"],
        decoder_name=ner_args["decoder_name"],
    )
    update_values(to_args=ner_template, from_args_list=[ner_args])
    runner = Conll(ner_template)
    runner.predict()
