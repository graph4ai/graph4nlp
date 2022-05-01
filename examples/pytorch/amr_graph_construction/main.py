from copy import deepcopy
from typing import Union
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from graph4nlp.pytorch.data.data import GraphData, to_batch

from graph4nlp.pytorch.models.graph2seq_loss import Graph2SeqLoss
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.data.dataset import Dataset, DataItem
from graph4nlp.pytorch.modules.graph_construction.base import (
    DynamicGraphConstructionBase,
    StaticGraphConstructionBase,
)
from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import (
    ConstituencyBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import (
    DependencyBasedGraphConstruction,
)

from amr_graph_construction import (
    AmrGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import (
    NodeEmbeddingBasedRefinedGraphConstruction,
)

from amr_semantic_parsing.graph2seq.args import get_args
from amr_semantic_parsing.graph2seq.build_model import get_model
from amr_semantic_parsing.graph2seq.evaluation import ExpressionAccuracy
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size
from amr_semantic_parsing.graph2seq.utils import get_log, wordid2str


class AmrDataItem(DataItem):
    def __init__(self, input_text, output_text, tokenizer, share_vocab=True):
        super(AmrDataItem, self).__init__(input_text, tokenizer)
        self.output_text = output_text
        self.share_vocab = share_vocab

    def extract(self):
        """
        Returns
        -------
        Input tokens and output tokens
        """
        g: GraphData = self.graph

        input_tokens = []
        for i in range(g.get_node_num()):
            if self.tokenizer is None:
                tokenized_token = g.node_attributes[i]["token"].strip().split(" ")
            else:
                tokenized_token = self.tokenizer(g.node_attributes[i]["token"])
            
            input_tokens.extend(tokenized_token)
        for s in g.graph_attributes["sentence"]:
            input_tokens.extend(s.strip().split(" "))

        if self.tokenizer is None:
            output_tokens = self.output_text.strip().split(" ")
        else:
            output_tokens = self.tokenizer(self.output_text)

        if self.share_vocab:
            return input_tokens + output_tokens
        else:
            return input_tokens, output_tokens

class Text2TextDataset(Dataset):
    """
        The dataset for text-to-text applications.
    Parameters
    ----------
    graph_construction_name: str
        The name of graph construction method. E.g., "dependency".
        Note that if it is in the provided graph names (i.e., "dependency", \
            "constituency", "ie", "node_emb", "node_emb_refine"), the following \
            parameters are set by default and users can't modify them:
            1. ``topology_builder``
            2. ``static_or_dynamic``
        If you need to customize your graph construction method, you should rename the \
            ``graph_construction_name`` and set the parameters above.
    root_dir: str, default=None
        The path of dataset.
    topology_builder: Union[StaticGraphConstructionBase, DynamicGraphConstructionBase], default=None
        The graph construction class.
    topology_subdir: str
        The directory name of processed path.
    static_or_dynamic: str, default='static'
        The graph type. Expected in ('static', 'dynamic')
    dynamic_init_graph_name: str, default=None
        The graph name of the initial graph. Expected in (None, "line", \
            "dependency", "constituency").
        Note that if it is in the provided graph names (i.e., "line", "dependency", \
            "constituency"), the following parameters are set by default and users \
            can't modify them:
            1. ``dynamic_init_topology_builder``
        If you need to customize your graph construction method, you should rename the \
            ``graph_construction_name`` and set the parameters above.
    dynamic_init_topology_builder: StaticGraphConstructionBase
        The graph construction class.
    dynamic_init_topology_aux_args: None,
        TBD.
    """

    def __init__(
        self,
        graph_construction_name: str,
        root_dir: str = None,
        static_or_dynamic: str = None,
        topology_builder: Union[
            StaticGraphConstructionBase, DynamicGraphConstructionBase
        ] = DependencyBasedGraphConstruction,
        topology_subdir: str = None,
        dynamic_init_graph_name: str = None,
        dynamic_init_topology_builder: StaticGraphConstructionBase = None,
        dynamic_init_topology_aux_args=None,
        share_vocab=True,
        **kwargs,
    ):
        if kwargs.get("graph_type", None) is not None:
            raise ValueError(
                "The argument ``graph_type`` is disgarded. \
                    Please use ``static_or_dynamic`` instead."
            )
        self.data_item_type = AmrDataItem
        self.share_vocab = share_vocab

        if graph_construction_name == "dependency":
            topology_builder = DependencyBasedGraphConstruction
            static_or_dynamic = "static"
        elif graph_construction_name == "amr":
            topology_builder = AmrGraphConstruction
            static_or_dynamic = "static"
        elif graph_construction_name == "constituency":
            topology_builder = ConstituencyBasedGraphConstruction
            static_or_dynamic = "static"
        elif graph_construction_name == "ie":
            topology_builder = IEBasedGraphConstruction
            static_or_dynamic = "static"
        elif graph_construction_name == "node_emb":
            topology_builder = NodeEmbeddingBasedGraphConstruction
            static_or_dynamic = "dynamic"
        elif graph_construction_name == "node_emb_refined":
            topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
            static_or_dynamic = "dynamic"
        else:
            print("Your are customizing the graph construction method.")
            if topology_builder is None:
                raise ValueError("``topology_builder`` can't be None if graph is defined by user.")
            if static_or_dynamic is None:
                raise ValueError("``static_or_dynamic`` can't be None if graph is defined by user.")

        if static_or_dynamic == "dynamic":
            if dynamic_init_graph_name is None or dynamic_init_graph_name == "line":
                dynamic_init_topology_builder = None
            elif dynamic_init_graph_name == "dependency":
                dynamic_init_topology_builder = DependencyBasedGraphConstruction
            elif dynamic_init_graph_name == "constituency":
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            else:
                if dynamic_init_topology_builder is None:
                    raise ValueError(
                        "``dynamic_init_topology_builder`` can't be None \
                            if ``dynamic_init_graph_name`` is defined by user."
                    )

        self.static_or_dynamic = static_or_dynamic
        super(Text2TextDataset, self).__init__(
            root=root_dir,
            graph_construction_name=graph_construction_name,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            static_or_dynamic=static_or_dynamic,
            share_vocab=share_vocab,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            **kwargs,
        )

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format is
        specified by each individual task-specific base class. Returns all
        the indices of data items in this file w.r.t. the whole dataset.

        For Text2TextDataset, the format of the input file should contain
        lines of input, each line representing one record of data. The
        input and output is separated by a tab(\t).

        Examples
        --------
        input: list job use languageid0 job ( ANS ) , language ( ANS , languageid0 )

        DataItem:
            input_text="list job use languageid0",
            output_text="job ( ANS ) , language ( ANS , languageid0 )"

        Parameters
        ----------
        file_path: str
            The path of the input file.

        Returns
        -------
        list
            The indices of data items in the file w.r.t. the whole dataset.
        """
        data = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                input, output = line.split("\t")
                data_item = AmrDataItem(
                    input_text=input,
                    output_text=output,
                    tokenizer=self.tokenizer,
                    share_vocab=self.share_vocab,
                )
                data.append(data_item)
        return data

    @classmethod
    def _vectorize_one_dataitem(cls, data_item, vocab_model, use_ie=False):

        item = deepcopy(data_item)
        graph: GraphData = item.graph
        token_matrix = []
        for node_idx in range(graph.get_node_num()):
            node_token = graph.node_attributes[node_idx]["token"]
            node_token_id = vocab_model.in_word_vocab.getIndex(node_token, use_ie)
            graph.node_attributes[node_idx]["token_id"] = node_token_id

            token_matrix.append([node_token_id])

        if "pos_tag" in graph.graph_attributes:
            pos_vocab = [".", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
            pos_map = {pos: i for i, pos in enumerate(pos_vocab)}
            maxlen = max(len(pos_tag) for pos_tag in graph.graph_attributes["pos_tag"])
            pos_token_id = torch.zeros(len(graph.graph_attributes["pos_tag"]), maxlen, dtype=torch.long)
            for i, sentence_token in enumerate(graph.graph_attributes["pos_tag"]):
                for j, token in enumerate(sentence_token):
                    if token in pos_map:
                        pos_token_id[i][j] = pos_map[token]
                    else:
                        print('pos_tag', token)
            graph.graph_attributes["pos_tag_id"] = pos_token_id 

        if "entity_label" in graph.graph_attributes:
            entity_label = ["O", "PERSON", "LOCATION", "ORGANIZATION", "ORGANIZATION", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "DATE", "TIME", "DURATION", "SET", "EMAIL", "URL", "CITY", "STATE_OR_PROVINCE", "COUNTRY", "NATIONALITY", "RELIGION", "TITLE", "IDEOLOGY", "CRIMINAL_CHARGE", "CAUSE_OF_DEATH", "HANDLE"]
            entity_map = {entity: i for i, entity in enumerate(entity_label)}
            maxlen = max(len(entity_tag) for entity_tag in graph.graph_attributes["entity_label"])
            entity_token_id = torch.zeros(len(graph.graph_attributes["entity_label"]), maxlen, dtype=torch.long)
            for i, sentence_token in enumerate(graph.graph_attributes["entity_label"]):
                for j, token in enumerate(sentence_token):
                    if token in entity_map:
                        entity_token_id[i][j] = entity_map[token]
                    else:
                        print('entity_label', token)
            graph.graph_attributes["entity_label_id"] = entity_token_id

        if "sentence" in graph.graph_attributes:
            maxlen = max(len(sentence.strip().split()) for sentence in graph.graph_attributes["sentence"])
            seq_token_id = torch.zeros(len(graph.graph_attributes["sentence"]), maxlen, dtype=torch.long)
            for i, sentence in enumerate(graph.graph_attributes["sentence"]):
                sentence_token = sentence.strip().split()
                for j, token in enumerate(sentence_token):
                    seq_token_id[i][j] = vocab_model.in_word_vocab.getIndex(token, use_ie)
            graph.graph_attributes["sentence_id"] = seq_token_id
                
        if use_ie:
            for i in range(len(token_matrix)):
                token_matrix[i] = np.array(token_matrix[i][0])
            token_matrix = pad_2d_vals_no_size(token_matrix)
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features["token_id"] = token_matrix
        else:
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features["token_id"] = token_matrix

        if use_ie and "token" in graph.edge_attributes[0].keys():
            edge_token_matrix = []
            for edge_idx in range(graph.get_edge_num()):
                edge_token = graph.edge_attributes[edge_idx]["token"]
                edge_token_id = vocab_model.in_word_vocab.getIndex(edge_token, use_ie)
                graph.edge_attributes[edge_idx]["token_id"] = edge_token_id
                edge_token_matrix.append([edge_token_id])
            if use_ie:
                for i in range(len(edge_token_matrix)):
                    edge_token_matrix[i] = np.array(edge_token_matrix[i][0])
                edge_token_matrix = pad_2d_vals_no_size(edge_token_matrix)
                edge_token_matrix = torch.tensor(edge_token_matrix, dtype=torch.long)
                graph.edge_features["token_id"] = edge_token_matrix

        tgt = item.output_text
        if isinstance(tgt, str):
            tgt_token_id = vocab_model.out_word_vocab.to_index_sequence(tgt)
            tgt_token_id.append(vocab_model.out_word_vocab.EOS)
            tgt_token_id = np.array(tgt_token_id)
            item.output_np = tgt_token_id
        return item

    def vectorization(self, data_items):
        if self.topology_builder == IEBasedGraphConstruction:
            use_ie = True
        else:
            use_ie = False
        for idx in range(len(data_items)):
            data_items[idx] = self._vectorize_one_dataitem(
                data_items[idx], self.vocab_model, use_ie=use_ie
            )

    @staticmethod
    def collate_fn(data_list: [AmrDataItem]):
        graph_list = [item.graph for item in data_list]
        graph_data = to_batch(graph_list)

        if isinstance(data_list[0].output_text, str):  # has ground truth
            output_numpy = [deepcopy(item.output_np) for item in data_list]
            output_str = [deepcopy(item.output_text.lower().strip()) for item in data_list]
            output_pad = pad_2d_vals_no_size(output_numpy)
            tgt_seq = torch.from_numpy(output_pad).long()
        else:
            output_str = []
            tgt_seq = None
        return {"graph_data": graph_data, "tgt_seq": tgt_seq, "output_str": output_str}

class JobsDataset(Text2TextDataset):
    """

        Parameters
        ----------
        root_dir: str
            The path of dataset.
        graph_construction_name: str
            The name of graph construction method. E.g., "dependency".
            Note that if it is in the provided graph names (i.e., "dependency", \
                "constituency", "ie", "node_emb", "node_emb_refine"), the following \
                parameters are set by default and users can't modify them:
                1. ``topology_builder``
                2. ``static_or_dynamic``
            If you need to customize your graph construction method, you should rename the \
                ``graph_construction_name`` and set the parameters above.
        topology_builder: GraphConstructionBase, default=None
            The graph construction class.
        topology_subdir: str
            The directory name of processed path.
        static_or_dynamic: str, default='static'
            The graph type. Expected in ('static', 'dynamic')
        edge_strategy: str, default=None
            The edge strategy. Expected in (None, 'homogeneous', 'as_node').
            If set `None`, it will be 'homogeneous'.
        merge_strategy: str, default=None
            The strategy to merge sub-graphs. Expected in (None, 'tailhead', 'user_define').
            If set `None`, it will be 'tailhead'.
        share_vocab: bool, default=False
            Whether to share the input vocabulary with the output vocabulary.
        dynamic_init_graph_name: str, default=None
            The graph name of the initial graph. Expected in (None, "line", "dependency", \
                "constituency").
            Note that if it is in the provided graph names (i.e., "line", "dependency", \
                "constituency"), the following parameters are set by default and users \
                can't modify them:
                1. ``dynamic_init_topology_builder``
            If you need to customize your graph construction method, you should rename the \
                ``graph_name`` and set the parameters above.
        dynamic_init_topology_builder: GraphConstructionBase
            The graph construction class.
        dynamic_init_topology_aux_args: None,
            TBD.
        """

    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {"train": "train.txt", "test": "test.txt"}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {"vocab": "vocab.pt", "data": "data.pt"}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded.
        # Please prepare the raw data yourself.')
        return

    def __init__(
        self,
        root_dir,
        topology_subdir,
        graph_construction_name,
        static_or_dynamic="static",
        topology_builder=None,
        merge_strategy="tailhead",
        edge_strategy=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
        pretrained_word_emb_name="6B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=None,
        seed=None,
        word_emb_size=300,
        share_vocab=True,
        lower_case=True,
        thread_number=1,
        port=9000,
        for_inference=None,
        reused_vocab_model=None,
    ):
        # Initialize the dataset. If the preprocessed files are not found,
        # then do the preprocessing and save them.
        super(JobsDataset, self).__init__(
            root_dir=root_dir,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            graph_construction_name=graph_construction_name,
            static_or_dynamic=static_or_dynamic,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            share_vocab=share_vocab,
            lower_case=lower_case,
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            seed=seed,
            word_emb_size=word_emb_size,
            thread_number=thread_number,
            port=port,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
        )

class Jobs:
    def __init__(self, opt):
        super(Jobs, self).__init__()
        self.opt = opt
        self.use_copy = self.opt["model_args"]["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_coverage = self.opt["model_args"]["decoder_args"]["rnn_decoder_share"][
            "use_coverage"
        ]
        self._build_device(self.opt)
        self._build_logger(self.opt["training_args"]["log_file"])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()
        self._build_loss_function()

    def _build_device(self, opt):
        seed = opt["env_args"]["seed"]
        np.random.seed(seed)
        if opt["env_args"]["use_gpu"] != 0 and torch.cuda.is_available():
            print("[ Using CUDA ]")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn

            cudnn.benchmark = True
            device = torch.device(
                "cuda" if opt["env_args"]["gpuid"] < 0 else "cuda:%d" % opt["env_args"]["gpuid"]
            )
        else:
            print("[ Using CPU ]")
            device = torch.device("cpu")
        self.device = device

    def _build_logger(self, log_file):
        import os

        log_folder = os.path.split(log_file)[0]
        if not os.path.exists(log_file):
            os.makedirs(log_folder)
        self.logger = get_log(log_file)

    def _build_dataloader(self):
        dataset = JobsDataset(
            root_dir=self.opt["model_args"]["graph_construction_args"]["graph_construction_share"][
                "root_dir"
            ],
            #   pretrained_word_emb_file=self.opt["pretrained_word_emb_file"],
            pretrained_word_emb_name=self.opt["preprocessing_args"]["pretrained_word_emb_name"],
            pretrained_word_emb_url=self.opt["preprocessing_args"]["pretrained_word_emb_url"],
            pretrained_word_emb_cache_dir=self.opt["preprocessing_args"][
                "pretrained_word_emb_cache_dir"
            ],
            #   val_split_ratio=self.opt["val_split_ratio"],
            merge_strategy=self.opt["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ]["merge_strategy"],
            edge_strategy=self.opt["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ]["edge_strategy"],
            seed=self.opt["env_args"]["seed"],
            word_emb_size=self.opt["preprocessing_args"]["word_emb_size"],
            share_vocab=self.opt["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["share_vocab"],
            graph_construction_name=self.opt["model_args"]["graph_construction_name"],
            dynamic_init_graph_name=self.opt["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ].get("dynamic_init_graph_type", None),
            topology_subdir=self.opt["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["topology_subdir"],
            thread_number=self.opt["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["thread_number"],
            port=self.opt["model_args"]["graph_construction_args"]["graph_construction_share"][
                "port"
            ],
        )

        self.train_dataloader = DataLoader(
            dataset.train,
            batch_size=self.opt["training_args"]["batch_size"],
            shuffle=True,
            num_workers=self.opt["training_args"]["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=self.opt["training_args"]["batch_size"],
            shuffle=False,
            num_workers=self.opt["training_args"]["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.vocab = dataset.vocab_model

    def _build_model(self):
        self.model = get_model(self.opt, vocab_model=self.vocab).to(self.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.opt["training_args"]["learning_rate"])

    def _build_evaluation(self):
        self.metrics = [ExpressionAccuracy()]

    def _build_loss_function(self):
        self.loss = Graph2SeqLoss(
            ignore_index=self.vocab.out_word_vocab.PAD,
            use_coverage=self.use_coverage,
            coverage_weight=self.opt["training_args"]["coverage_weight"],
        )

    def train(self):
        max_score = -1
        self._best_epoch = -1
        for epoch in range(200):
            self.model.train()
            self.train_epoch(epoch, split="train")
            self._adjust_lr(epoch)
            if epoch >= 0:
                score = self.evaluate(split="test")
                if score >= max_score:
                    self.logger.info("Best model saved, epoch {}".format(epoch))
                    self.model.save_checkpoint(
                        self.opt["checkpoint_args"]["checkpoint_save_path"], "best.pt"
                    )
                    self._best_epoch = epoch
                max_score = max(max_score, score)
            if epoch >= 30 and self._stop_condition(epoch):
                break
        return max_score

    def _stop_condition(self, epoch, patience=20):
        return epoch > patience + self._best_epoch

    def _adjust_lr(self, epoch):
        def set_lr(optimizer, decay_factor):
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * decay_factor

        epoch_diff = epoch - self.opt["training_args"]["lr_start_decay_epoch"]
        if epoch_diff >= 0 and epoch_diff % self.opt["training_args"]["lr_decay_per_epoch"] == 0:
            if self.opt["training_args"]["learning_rate"] > self.opt["training_args"]["min_lr"]:
                set_lr(self.optimizer, self.opt["training_args"]["lr_decay_rate"])
                self.opt["training_args"]["learning_rate"] = (
                    self.opt["training_args"]["learning_rate"]
                    * self.opt["training_args"]["lr_decay_rate"]
                )
                self.logger.info(
                    "Learning rate adjusted: {:.5f}".format(
                        self.opt["training_args"]["learning_rate"]
                    )
                )

    def train_epoch(self, epoch, split="train"):
        assert split in ["train"]
        self.logger.info("Start training in split {}, Epoch: {}".format(split, epoch))
        loss_collect = []
        dataloader = self.train_dataloader
        step_all_train = len(dataloader)
        for step, data in enumerate(dataloader):
            graph, tgt, gt_str = data["graph_data"], data["tgt_seq"], data["output_str"]
            graph = graph.to(self.device)
            tgt = tgt.to(self.device)
            oov_dict = None
            if self.use_copy:
                oov_dict, tgt = prepare_ext_vocab(
                    graph, self.vocab, gt_str=gt_str, device=self.device
                )

            prob, enc_attn_weights, coverage_vectors = self.model(graph, tgt, oov_dict=oov_dict)
            loss = self.loss(
                logits=prob,
                label=tgt,
                enc_attn_weights=enc_attn_weights,
                coverage_vectors=coverage_vectors,
            )
            loss_collect.append(loss.item())
            if step % self.opt["training_args"]["loss_display_step"] == 0 and step != 0:
                self.logger.info(
                    "Epoch {}: [{} / {}] loss: {:.3f}".format(
                        epoch, step, step_all_train, np.mean(loss_collect)
                    )
                )
                loss_collect = []
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, split="val"):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["test"]
        dataloader = self.test_dataloader
        for data in dataloader:
            graph, gt_str = data["graph_data"], data["output_str"]
            graph = graph.to(self.device)
            if self.use_copy:
                oov_dict = prepare_ext_vocab(
                    batch_graph=graph, vocab=self.vocab, device=self.device
                )
                ref_dict = oov_dict
            else:
                oov_dict = None
                ref_dict = self.vocab.out_word_vocab

            prob, _, _ = self.model(graph, oov_dict=oov_dict)
            pred = prob.argmax(dim=-1)

            pred_str = wordid2str(pred.detach().cpu(), ref_dict)
            pred_collect.extend(pred_str)
            gt_collect.extend(gt_str)

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation accuracy in `{}` split: {:.3f}".format(split, score))
        return score

    @torch.no_grad()
    def translate(self):
        self.model.eval()

        pred_collect = []
        gt_collect = []
        dataloader = self.test_dataloader
        for data in dataloader:
            graph, gt_str = data["graph_data"], data["output_str"]
            graph = graph.to(self.device)
            if self.use_copy:
                oov_dict = prepare_ext_vocab(
                    batch_graph=graph, vocab=self.vocab, device=self.device
                )
                ref_dict = oov_dict
            else:
                oov_dict = None
                ref_dict = self.vocab.out_word_vocab

            pred = self.model.translate(batch_graph=graph, oov_dict=oov_dict, beam_size=4, topk=1)

            pred_ids = pred[:, 0, :]  # we just use the top-1

            pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)

            pred_collect.extend(pred_str)
            gt_collect.extend(gt_str)

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation accuracy in `{}` split: {:.3f}".format("test", score))
        return score


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->  {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    opt = get_args()
    config = load_json_config(opt["json_config"])
    print_config(config)
    print(config.keys())

    runner = Jobs(config)
    max_score = runner.train()
    runner.logger.info("Train finish, best val score: {:.3f}".format(max_score))
    runner.translate()
