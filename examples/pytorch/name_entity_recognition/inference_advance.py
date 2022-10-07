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
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config
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
    def __init__(self, config):
        super(Conll, self).__init__()
        self.tag_types = ["I-PER", "O", "B-ORG", "B-LOC", "I-ORG", "I-MISC", "I-LOC", "B-MISC"]
        if config["env_args"]["gpu"] > -1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.checkpoint_save_path = "./checkpoints/"
        self.config = config
        print("finish building model")
        self._build_model()

        print("finish dataloading")
        self._build_dataloader()

        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        print("starting build the dataset")
        self.graph_name = self.config["model_args"]["graph_construction_name"]

        if self.graph_name == "line_graph":
            dataset = ConllDataset(
                root_dir="examples/pytorch/name_entity_recognition/conll",
                topology_builder=LineBasedGraphConstruction,
                graph_name=self.graph_name,
                static_or_dynamic="static",
                pretrained_word_emb_cache_dir=self.config["model_args"]["graph_construction_args"][
                    "graph_construction_share"
                ]["pre_word_emb_file"],
                topology_subdir="LineGraph",
                tag_types=self.tag_types,
                for_inference=1,
                reused_vocab_model=self.model.vocab,
            )
        elif self.graph_name == "dependency_graph":
            dataset = ConllDataset(
                root_dir=self.config["model_args"]["graph_construction_args"][
                    "graph_construction_share"
                ]["root_dir"],
                topology_builder=DependencyBasedGraphConstruction_without_tokenizer,
                graph_name=self.graph_name,
                static_or_dynamic="static",
                pretrained_word_emb_cache_dir=self.config["model_args"]["graph_construction_args"][
                    "graph_construction_share"
                ]["pre_word_emb_file"],
                topology_subdir="DependencyGraph",
                tag_types=self.tag_types,
                nlp_processor_args=self.config["model_args"]["graph_construction_args"][
                    "nlp_processor_args"
                ],
            )
        elif self.graph_name == "node_emb":
            dataset = ConllDataset(
                root_dir="examples/pytorch/name_entity_recognition/conll",
                topology_builder=NodeEmbeddingBasedGraphConstruction,
                graph_name=self.graph_name,
                pretrained_word_emb_cache_dir=self.config["model_args"]["graph_construction_args"][
                    "graph_construction_share"
                ]["pre_word_emb_file"],
                topology_subdir="DynamicGraph_node_emb",
                tag_types=self.tag_types,
                merge_strategy=None,
                for_inference=1,
                reused_vocab_model=self.model.vocab,
            )
        elif self.graph_name == "node_emb_refined":
            if (
                self.config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ].get("dynamic_init_graph_name", None)
                == "line"
            ):
                dynamic_init_topology_builder = LineBasedGraphConstruction
            elif (
                self.config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ].get("dynamic_init_graph_name", None)
                == "dependency"
            ):
                dynamic_init_topology_builder = DependencyBasedGraphConstruction_without_tokenizer
            else:
                # init_topology_builder
                raise RuntimeError("Define your own init_topology_builder")
            dataset = ConllDataset(
                root_dir="examples/pytorch/name_entity_recognition/conll",
                topology_builder=NodeEmbeddingBasedRefinedGraphConstruction,
                graph_name=self.graph_name,
                static_or_dynamic="dynamic",
                pretrained_word_emb_cache_dir=self.config["model_args"]["graph_construction_args"][
                    "graph_construction_share"
                ]["pre_word_emb_file"],
                topology_subdir="DynamicGraph_node_emb_refined",
                tag_types=self.tag_types,
                dynamic_init_graph_name=self.config["model_args"]["graph_construction_args"][
                    "graph_construction_private"
                ].get("dynamic_init_graph_name", None),
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
        self.optimizer = optim.Adam(
            parameters,
            lr=self.config["training_args"]["lr"],
            weight_decay=self.config["training_args"]["weight_decay"],
        )

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

    import datetime

    starttime = datetime.datetime.now()

    cfg = get_args()
    config = load_json_config(cfg["json_config"])
    print_config(config)

    runner = Conll(config)
    max_score = runner.test()
    print("Test finish, best score: {:.3f}".format(max_score))
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
