import numpy as np
import torch
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.jobs import JobsDataset
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import (
    ConstituencyBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import (
    DependencyBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import (  # noqa
    NodeEmbeddingBasedRefinedGraphConstruction,
)
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab

from args import get_args
from evaluation import ExpressionAccuracy
from utils import get_log, wordid2str


class Jobs:
    def __init__(self, opt):
        super(Jobs, self).__init__()
        self.opt = opt
        self._build_device(self.opt)
        self._build_model()

    def _build_device(self, opt):
        seed = opt["seed"]
        np.random.seed(seed)
        if opt["use_gpu"] != 0 and torch.cuda.is_available():
            print("[ Using CUDA ]")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn

            cudnn.benchmark = True
            device = torch.device("cuda" if opt["gpu"] < 0 else "cuda:%d" % opt["gpu"])
        else:
            print("[ Using CPU ]")
            device = torch.device("cpu")
        self.device = device

    def _build_model(self):
        self.model = Graph2Seq.load_checkpoint(self.opt["checkpoint_save_path"], "best.pt").to(
            self.device
        )

    @torch.no_grad()
    def translate(self):
        self.model.eval()
        from nltk.tokenize import word_tokenize
        print(self.opt)
        ret = self.model.predict(raw_sentences=["list job on platformid0"], beam_size=3, use_copy=self.opt["decoder_args"]["rnn_decoder_share"]["use_copy"],
                            tokenizer=word_tokenize, merge_strategy=self.opt["graph_construction_args"]["graph_construction_private"]["merge_strategy"], 
                            edge_strategy=self.opt["graph_construction_args"]["graph_construction_private"]["edge_strategy"])
        print(ret)


if __name__ == "__main__":
    opt = get_args()
    runner = Jobs(opt)
    runner.translate()
