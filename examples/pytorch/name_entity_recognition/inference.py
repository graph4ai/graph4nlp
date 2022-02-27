import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing

from graph4nlp.examples.pytorch.name_entity_recognition.conll import ConllDataset_inference
from graph4nlp.examples.pytorch.name_entity_recognition.conlleval import evaluate
from graph4nlp.examples.pytorch.name_entity_recognition.line_graph_construction import (
    LineBasedGraphConstruction,
)
from graph4nlp.examples.pytorch.name_entity_recognition.model import Word2tag
from graph4nlp.pytorch.data.dataset import SequenceLabelingDataItem
from graph4nlp.pytorch.inference_wrapper.classifier_inference_wrapper import (
    ClassifierInferenceWrapper,
)
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config
from graph4nlp.pytorch.modules.utils.generic_utils import to_cuda

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

        print("finish building model")
        self.opt = config
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
    # long running
    # do something other

    # args = parser.parse_args()
    # print("load ner template config")
    # ner_args = get_yaml_config(args.task_config)

    # ner_template = get_basic_args(
    #    graph_construction_name=ner_args["graph_construction_name"],
    #    graph_embedding_name=ner_args["graph_embedding_name"],
    #    decoder_name=ner_args["decoder_name"],
    # )
    # update_values(to_args=ner_template, from_args_list=[ner_args])
    cfg = get_args()
    config = load_json_config(cfg["json_config"])
    print_config(config)
    runner = Conll(config)
    runner.predict()
