import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader

from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy
from graph4nlp.pytorch.modules.graph_construction import NodeEmbeddingBasedRefinedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda

from conll import ConllDataset
from conlleval import evaluate
from dependency_graph_construction_without_tokenize import (
    DependencyBasedGraphConstruction_without_tokenizer,
)
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
    def __init__(self):
        super(Conll, self).__init__()
        self.tag_types = ["I-PER", "O", "B-ORG", "B-LOC", "I-ORG", "I-MISC", "I-LOC", "B-MISC"]
        if args.gpu > -1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.checkpoint_save_path = "./checkpoints/"

        print("finish building model")
        self._build_model()

        print("finish dataloading")
        self._build_dataloader()

        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        print("starting build the dataset")

        if args.graph_name == "line_graph":
            dataset = ConllDataset(
                root_dir="examples/pytorch/name_entity_recognition/conll",
                topology_builder=LineBasedGraphConstruction,
                graph_name=args.graph_name,
                static_or_dynamic=args.static_or_dynamic,
                pretrained_word_emb_cache_dir=args.pre_word_emb_file,
                topology_subdir="LineGraph",
                tag_types=self.tag_types,
                for_inference=1,
                reused_vocab_model=self.model.vocab,
            )
        elif args.graph_name == "dependency_graph":
            dataset = ConllDataset(
                root_dir="examples/pytorch/name_entity_recognition/conll",
                topology_builder=DependencyBasedGraphConstruction_without_tokenizer,
                graph_name=args.graph_name,
                pretrained_word_emb_cache_dir=args.pre_word_emb_file,
                topology_subdir="DependencyGraph",
                tag_types=self.tag_types,
                for_inference=1,
                reused_vocab_model=self.model.vocab,
            )
        elif args.graph_name == "node_emb":
            dataset = ConllDataset(
                root_dir="examples/pytorch/name_entity_recognition/conll",
                topology_builder=NodeEmbeddingBasedGraphConstruction,
                graph_name=args.graph_name,
                pretrained_word_emb_cache_dir=args.pre_word_emb_file,
                topology_subdir="DynamicGraph_node_emb",
                tag_types=self.tag_types,
                merge_strategy=None,
                for_inference=1,
                reused_vocab_model=self.model.vocab,
            )
        elif args.graph_name == "node_emb_refined":
            if args.init_graph_name == "line":
                dynamic_init_topology_builder = LineBasedGraphConstruction
            elif args.init_graph_name == "dependency":
                dynamic_init_topology_builder = DependencyBasedGraphConstruction_without_tokenizer
            else:
                # init_topology_builder
                raise RuntimeError("Define your own init_topology_builder")
            dataset = ConllDataset(
                root_dir="examples/pytorch/name_entity_recognition/conll",
                topology_builder=NodeEmbeddingBasedRefinedGraphConstruction,
                graph_name=args.graph_name,
                static_or_dynamic="dynamic",
                pretrained_word_emb_cache_dir=args.pre_word_emb_file,
                topology_subdir="DynamicGraph_node_emb_refined",
                tag_types=self.tag_types,
                dynamic_init_graph_name=args.init_graph_name,
                dynamic_init_topology_builder=dynamic_init_topology_builder,
                dynamic_init_topology_aux_args={"dummy_param": 0},
                for_inference=1,
                reused_vocab_model=self.model.vocab,
            )

        print("strating loading the testing data")
        self.test_dataloader = DataLoader(
            dataset.test, batch_size=100, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn
        )
        print("strating loading the vocab")
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = Word2tag.load_checkpoint(self.checkpoint_save_path, "best.pt").to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    def _build_evaluation(self):
        self.metrics = Accuracy(["F1", "precision", "recall"])

    def test(self):

        print("sucessfully loaded the existing saved model!")
        self.model.eval()
        pred_collect = []
        tokens_collect = []
        tgt_collect = []
        with torch.no_grad():
            for data in self.test_dataloader:
                graph, tgt = data["graph_data"], data["tgt_tag"]
                graph = graph.to(self.device)
                tgt_l = [tgt_.to(self.device) for tgt_ in tgt]
                pred, loss = self.model(graph, tgt_l, require_loss=True)
                # pred = logits2tag(g)
                pred_collect.extend(pred)
                tgt_collect.extend(tgt)
                tokens_collect.extend(get_tokens(from_batch(graph)))
        prec, rec, f1 = conll_score(pred_collect, tgt_collect, self.tag_types)
        print("Testing results: precision is %5.2f, rec is %5.2f, f1 is %5.2f" % (prec, rec, f1))
        return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
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
        help="graph_name:line_graph, dependency_graph, node_emb_graph",
    )
    parser.add_argument(
        "--static_or_dynamic",
        type=str,
        default="static",
        help="static or dynamic",
    )
    parser.add_argument(
        "--init_graph_name",
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

    import datetime

    starttime = datetime.datetime.now()
    # long running
    # do something other

    args = parser.parse_args()
    runner = Conll()
    max_score = runner.test()
    print("Test finish, best score: {:.3f}".format(max_score))
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
