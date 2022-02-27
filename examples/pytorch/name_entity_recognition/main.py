import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader

from graph4nlp.examples.pytorch.name_entity_recognition.conll import ConllDataset
from graph4nlp.examples.pytorch.name_entity_recognition.conlleval import evaluate
from graph4nlp.examples.pytorch.name_entity_recognition.line_graph_construction import (
    LineBasedGraphConstruction,
)
from graph4nlp.examples.pytorch.name_entity_recognition.model import Word2tag
from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy
from graph4nlp.pytorch.modules.graph_construction import NodeEmbeddingBasedRefinedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config
from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda

from dependency_graph_construction_without_tokenize import (
    DependencyBasedGraphConstruction_without_tokenizer,
)

torch.multiprocessing.set_sharing_strategy("file_system")
cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from torchcrf import CRF


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
        self.config = config
        if config["env_args"]["gpu"] > -1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.checkpoint_path = "./checkpoints/"
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self._build_dataloader()
        print("finish dataloading")
        self._build_model()
        print("finish building model")
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        print("starting build the dataset")

        self.graph_name = self.config["model_args"]["graph_construction_name"]

        if self.graph_name == "line_graph":
            dataset = ConllDataset(
                root_dir=self.config["model_args"]["graph_construction_args"][
                    "graph_construction_share"
                ][
                    "root_dir"
                ],  # "examples/pytorch/name_entity_recognition/conll",
                topology_builder=LineBasedGraphConstruction,
                graph_name=self.graph_name,
                static_or_dynamic="static",
                pretrained_word_emb_cache_dir=self.config["model_args"]["graph_construction_args"][
                    "graph_construction_share"
                ]["pre_word_emb_file"],
                topology_subdir="LineGraph",
                tag_types=self.tag_types,
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
            )
        elif self.graph_name == "node_emb":
            dataset = ConllDataset(
                root_dir=self.config["model_args"]["graph_construction_args"][
                    "graph_construction_share"
                ]["root_dir"],
                topology_builder=NodeEmbeddingBasedGraphConstruction,
                graph_name=self.graph_name,
                static_or_dynamic="static",
                pretrained_word_emb_cache_dir=self.config["model_args"]["graph_construction_args"][
                    "graph_construction_share"
                ]["pre_word_emb_file"],
                topology_subdir="DynamicGraph_node_emb",
                tag_types=self.tag_types,
                merge_strategy=None,
            )
        elif self.graph_type == "node_emb_refined":
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
                dynamic_init_topology_builder=dynamic_init_topology_builder,
                dynamic_init_topology_aux_args={"dummy_param": 0},
            )

        print(len(dataset.train))
        print("strating loading the training data")
        self.train_dataloader = DataLoader(
            dataset.train,
            batch_size=self.config["training_args"]["batch_size"],
            shuffle=True,
            num_workers=1,
            collate_fn=dataset.collate_fn,
        )
        print("strating loading the validating data")
        self.val_dataloader = DataLoader(
            dataset.val, batch_size=100, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn
        )
        print("strating loading the testing data")
        self.test_dataloader = DataLoader(
            dataset.test, batch_size=100, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn
        )
        print("strating loading the vocab")
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = Word2tag(self.vocab, self.config, device=self.device).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            parameters,
            lr=self.config["training_args"]["lr"],
            weight_decay=self.config["training_args"]["weight_decay"],
        )

    def _build_evaluation(self):
        self.metrics = Accuracy(["F1", "precision", "recall"])

    def train(self):
        max_score = -1
        max_idx = 0
        for epoch in range(self.config["training_args"]["epochs"]):
            self.model.train()
            print("Epoch: {}".format(epoch))
            pred_collect = []
            gt_collect = []
            for data in self.train_dataloader:
                graph, tgt = data["graph_data"], data["tgt_tag"]
                tgt_l = [tgt_.to(self.device) for tgt_ in tgt]
                graph = graph.to(self.device)
                pred_tags, loss = self.model(graph, tgt_l, require_loss=True)
                pred_collect.extend(pred_tags)  # pred: list of batch_sentence pred tensor
                gt_collect.extend(tgt)  # tgt:list of sentence token tensor
                # num_tokens=len(torch.cat(pred_tags).view(-1))
                print("Epoch: {}".format(epoch) + " loss:" + str(loss.cpu().item()))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            if epoch % 1 == 0:
                score = self.evaluate(epoch)
                if score > max_score:
                    self.model.save_checkpoint(self.checkpoint_path, "best.pt")
                    max_idx = epoch
                max_score = max(max_score, score)
        return max_score, max_idx

    def evaluate(self, epoch):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        tokens_collect = []
        with torch.no_grad():
            for data in self.val_dataloader:
                graph, tgt = data["graph_data"], data["tgt_tag"]
                graph = graph.to(self.device)
                tgt_l = [tgt_.to(self.device) for tgt_ in tgt]
                pred, loss = self.model(graph, tgt_l, require_loss=True)
                pred_collect.extend(pred)  # pred: list of batch_sentence pred tensor
                gt_collect.extend(tgt)  # tgt:list of sentence token tensor
                tokens_collect.extend(get_tokens(from_batch(graph)))

        prec, rec, f1 = conll_score(pred_collect, gt_collect, self.tag_types)
        print("Testing results: precision is %5.2f, rec is %5.2f, f1 is %5.2f" % (prec, rec, f1))
        print("Epoch: {}".format(epoch) + " loss:" + str(loss.cpu().item()))

        return f1

    @torch.no_grad()
    def test(self):
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

    import datetime

    starttime = datetime.datetime.now()
    cfg = get_args()
    config = load_json_config(cfg["json_config"])
    print_config(config)
    runner = Conll(config)
    max_score, max_idx = runner.train()
    print("Train finish, best score: {:.3f}".format(max_score))
    print(max_idx)
    score = runner.test()
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
