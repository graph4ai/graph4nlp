import argparse
import copy
import multiprocessing
import os
import platform
import time
from typing import Union
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
import nltk
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from graph4nlp.pytorch.data.data import GraphData, to_batch
from copy import deepcopy

from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.models.graph2seq_loss import Graph2SeqLoss
from graph4nlp.pytorch.modules.evaluation import BLEU, METEOR, ROUGE
from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import (
    WordEmbedding,
)
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.config_utils import load_json_config
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.utils.generic_utils import EarlyStopping, to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size

from .question_generation.fused_embedding_construction import FusedEmbeddingConstruction


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

from .amr_graph_construction import (
    AmrGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_construction.node_embedding_based_refined_graph_construction import (
    NodeEmbeddingBasedRefinedGraphConstruction,
)

class AmrDataItem(DataItem):  
    def __init__(self, input_text, input_text2, output_text, tokenizer, share_vocab=True):
        super(AmrDataItem, self).__init__(input_text, tokenizer)
        self.input_text2 = input_text2
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
                tokenized_token = g.node_attributes[i]["token"].strip().split()
            else:
                tokenized_token = self.tokenizer(g.node_attributes[i]["token"])

            input_tokens.extend(tokenized_token)
        
        for s in g.graph_attributes["sentence"]:
            input_tokens.extend(s.strip().split())

        if self.tokenizer is None:
            input_tokens.extend(self.input_text2.strip().split())
            output_tokens = self.output_text.strip().split()
        else:
            input_tokens.extend(self.tokenizer(self.input_text2))
            output_tokens = self.tokenizer(self.output_text)

        if self.share_vocab:
            return input_tokens + output_tokens
        else:
            return input_tokens, output_tokens
    

class DoubleText2TextDataset(Dataset):
    """
        The dataset for double-text-to-text applications.
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
        elif graph_construction_name == "constituency":
            topology_builder = ConstituencyBasedGraphConstruction
            static_or_dynamic = "static"
        elif graph_construction_name == "ie":
            topology_builder = IEBasedGraphConstruction
            static_or_dynamic = "static"
        elif graph_construction_name == "amr":
            topology_builder = AmrGraphConstruction
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
        super(DoubleText2TextDataset, self).__init__(
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
        specified by each individual task-specific base class.
        Returns all the indices of data items in this file w.r.t. the whole dataset.

        For DoubleText2TextDataset, the format of the input file should
        contain lines of input, each line representing one record of data.
        The input and output is separated by a tab(\t).
        # TODO: update example

        Examples
        --------
        input: list job use languageid0 job ( ANS ) , language ( ANS , languageid0 )

        DataItem:
            input_text="list job use languageid0",
            input_text2="list job use languageid0",
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
            for line in f:
                if self.lower_case:
                    line = line.lower().strip()
                input, input2, output = line.split("\t")
                data_item = AmrDataItem(
                    input_text=input.strip(),
                    input_text2=input2.strip(),
                    output_text=output.strip(),
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

            for i in range(len(edge_token_matrix)):
                edge_token_matrix[i] = np.array(edge_token_matrix[i][0])

            edge_token_matrix = pad_2d_vals_no_size(edge_token_matrix)
            edge_token_matrix = torch.tensor(edge_token_matrix, dtype=torch.long)
            graph.edge_features["token_id"] = edge_token_matrix

        item.input_text2 = vocab_model.tokenizer(item.input_text2)
        input_token_id2 = vocab_model.in_word_vocab.to_index_sequence_for_list(item.input_text2)
        input_token_id2 = np.array(input_token_id2)
        item.input_np2 = input_token_id2

        if isinstance(item.output_text, str):
            if vocab_model.in_word_vocab.lower_case:
                item.output_text = item.output_text.lower()

            item.output_text = vocab_model.tokenizer(item.output_text)
            tgt_token_id = vocab_model.in_word_vocab.to_index_sequence_for_list(item.output_text)
            tgt_token_id.append(vocab_model.in_word_vocab.EOS)
            tgt_token_id = np.array(tgt_token_id)
            item.output_np = tgt_token_id
            item.output_text = " ".join(item.output_text)
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
        graph_data = []
        input_tensor2, input_length2, input_text2 = [], [], []
        tgt_tensor, tgt_text = [], []
        for item in data_list:
            graph_data.append(item.graph)
            input_tensor2.append(deepcopy(item.input_np2))
            input_length2.append(len(item.input_np2))
            input_text2.append(deepcopy(item.input_text2))

            if isinstance(item.output_text, str):
                tgt_tensor.append(deepcopy(item.output_np))
                tgt_text.append(deepcopy(item.output_text))

        input_tensor2 = torch.LongTensor(pad_2d_vals_no_size(input_tensor2))
        input_length2 = torch.LongTensor(input_length2)
        graph_data = to_batch(graph_data)
        if len(tgt_tensor) > 0:
            tgt_tensor = torch.LongTensor(pad_2d_vals_no_size(tgt_tensor))

        return {
            "graph_data": graph_data,
            "input_tensor2": input_tensor2,
            "input_text2": input_text2,
            "tgt_tensor": tgt_tensor,
            "tgt_text": tgt_text,
            "input_length2": input_length2,
        }

class SQuADDataset(DoubleText2TextDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        if self.for_inference:
            return {"test": "test.txt"}
        else:
            return {"train": "train.txt", "val": "val.txt", "test": "test.txt"}

    @property
    def processed_file_names(self):
        """At least 2 reserved keys should be fiiled: 'vocab' and 'data'."""
        return {"vocab": "vocab.pt", "data": "data.pt"}

    def download(self):
        raise NotImplementedError(
            "This dataset is now under test and cannot be downloaded."
            "Please prepare the raw data yourself."
        )

    def __init__(
        self,
        root_dir,
        topology_subdir,
        graph_construction_name,
        static_or_dynamic="static",
        topology_builder=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
        share_vocab=True,
        pretrained_word_emb_name="840B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=None,
        max_word_vocab_size=None,
        min_word_vocab_freq=1,
        tokenizer=nltk.RegexpTokenizer(" ", gaps=True).tokenize,
        word_emb_size=None,
        **kwargs
    ):
        super(SQuADDataset, self).__init__(
            graph_construction_name,
            root_dir=root_dir,
            static_or_dynamic=static_or_dynamic,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            share_vocab=share_vocab,
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            max_word_vocab_size=max_word_vocab_size,
            min_word_vocab_freq=min_word_vocab_freq,
            tokenizer=tokenizer,
            word_emb_size=word_emb_size,
            **kwargs
        )

class QGModel(nn.Module):
    def __init__(self, vocab, config):
        super(QGModel, self).__init__()
        self.config = config
        self.vocab = vocab
        self.use_coverage = self.config["model_args"]["decoder_args"]["rnn_decoder_share"][
            "use_coverage"
        ]

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
                fix_emb=config["model_args"]["graph_initialization_args"]["fix_word_emb"],
            ).word_emb_layer
        self.g2s.seq_decoder.tgt_emb = self.word_emb

        self.loss_calc = Graph2SeqLoss(
            ignore_index=self.vocab.out_word_vocab.PAD,
            use_coverage=self.use_coverage,
            coverage_weight=config["training_args"]["coverage_loss_ratio"],
        )

        # Replace the default embedding construction layer
        #   with the customized passage-answer alignment embedding construction layer
        # TODO: delete the default layer and clear the memory
        embedding_styles = config["model_args"]["graph_initialization_args"]["embedding_style"]
        self.g2s.graph_initializer.embedding_layer = FusedEmbeddingConstruction(
            self.vocab.in_word_vocab,
            embedding_styles["single_token_item"],
            emb_strategy=embedding_styles["emb_strategy"],
            hidden_size=config["model_args"]["graph_initialization_args"]["hidden_size"],
            num_rnn_layers=embedding_styles.get("num_rnn_layers", 1),
            fix_word_emb=config["model_args"]["graph_initialization_args"]["fix_word_emb"],
            fix_bert_emb=config["model_args"]["graph_initialization_args"]["fix_bert_emb"],
            bert_model_name=embedding_styles.get("bert_model_name", "bert-base-uncased"),
            bert_lower_case=embedding_styles.get("bert_lower_case", True),
            word_dropout=config["model_args"]["graph_initialization_args"]["word_dropout"],
            bert_dropout=config["model_args"]["graph_initialization_args"].get(
                "bert_dropout", None
            ),
            rnn_dropout=config["model_args"]["graph_initialization_args"]["rnn_dropout"],
        )
        self.graph_construction_name = self.g2s.graph_construction_name
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
        self.use_copy = self.config["model_args"]["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_coverage = self.config["model_args"]["decoder_args"]["rnn_decoder_share"][
            "use_coverage"
        ]
        log_config = copy.deepcopy(config)
        del log_config["env_args"]["device"]
        self.logger = Logger(
            config["checkpoint_args"]["out_dir"],
            config=log_config,
            overwrite=True,
        )
        self.logger.write(config["checkpoint_args"]["out_dir"])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

    def _build_dataloader(self):
        for i in self.config["model_args"]["graph_construction_args"]:
            print(i)
        dataset = SQuADDataset(
            root_dir=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["root_dir"],
            topology_subdir=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["topology_subdir"],
            graph_construction_name=self.config["model_args"]["graph_construction_name"],
            dynamic_init_graph_name=self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ].get("dynamic_init_graph_type", None),
            dynamic_init_topology_aux_args={"dummy_param": 0},
            pretrained_word_emb_name=self.config["preprocessing_args"]["pretrained_word_emb_name"],
            merge_strategy=self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ]["merge_strategy"],
            edge_strategy=self.config["model_args"]["graph_construction_args"][
                "graph_construction_private"
            ]["edge_strategy"],
            max_word_vocab_size=self.config["preprocessing_args"]["top_word_vocab"],
            min_word_vocab_freq=self.config["preprocessing_args"]["min_word_freq"],
            word_emb_size=self.config["preprocessing_args"]["word_emb_size"],
            share_vocab=self.config["preprocessing_args"]["share_vocab"],
            seed=self.config["env_args"]["seed"],
            thread_number=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["thread_number"],
            port=self.config["model_args"]["graph_construction_args"]["graph_construction_share"][
                "port"
            ],
            timeout=self.config["model_args"]["graph_construction_args"][
                "graph_construction_share"
            ]["timeout"],
        )

        self.train_dataloader = DataLoader(
            dataset.train,
            batch_size=self.config["training_args"]["batch_size"],
            shuffle=True,
            num_workers=self.config["env_args"]["num_workers"],
            collate_fn=dataset.collate_fn,
        )
        self.val_dataloader = DataLoader(
            dataset.val,
            batch_size=self.config["training_args"]["batch_size"],
            shuffle=False,
            num_workers=self.config["env_args"]["num_workers"],
            collate_fn=dataset.collate_fn,
        )
        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=self.config["training_args"]["batch_size"],
            shuffle=False,
            num_workers=self.config["env_args"]["num_workers"],
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
        self.model = QGModel(self.vocab, self.config).to(self.config["env_args"]["device"])

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.config["training_args"]["lr"])
        self.stopper = EarlyStopping(
            os.path.join(self.config["checkpoint_args"]["out_dir"], Constants._SAVED_WEIGHTS_FILE),
            patience=self.config["training_args"]["patience"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config["training_args"]["lr_reduce_factor"],
            patience=self.config["training_args"]["lr_patience"],
            verbose=True,
        )

    def _build_evaluation(self):
        self.metrics = {"BLEU": BLEU(n_grams=[1, 2, 3, 4]), "METEOR": METEOR(), "ROUGE": ROUGE()}

    def train(self):
        for epoch in range(self.config["training_args"]["epochs"]):
            self.model.train()
            train_loss = []
            dur = []
            t0 = time.time()
            for i, data in enumerate(self.train_dataloader):
                data = all_to_cuda(data, self.config["env_args"]["device"])
                data["graph_data"] = data["graph_data"].to(self.config["env_args"]["device"])
                oov_dict = None
                if self.use_copy:
                    oov_dict, tgt = prepare_ext_vocab(
                        data["graph_data"],
                        self.vocab,
                        gt_str=data["tgt_text"],
                        device=self.config["env_args"]["device"],
                    )
                    data["tgt_tensor"] = tgt

                logits, loss = self.model(data, oov_dict=oov_dict, require_loss=True)
                self.optimizer.zero_grad()
                loss.backward()
                if self.config["training_args"].get("grad_clipping", None) not in (None, 0):
                    # Clip gradients
                    parameters = [p for p in self.model.parameters() if p.requires_grad]

                    torch.nn.utils.clip_grad_norm_(
                        parameters, self.config["training_args"]["grad_clipping"]
                    )

                self.optimizer.step()
                train_loss.append(loss.item())

                # pred = torch.max(logits, dim=-1)[1].cpu()
                dur.append(time.time() - t0)
                if (i + 1) % 100 == 0:
                    format_str = (
                        "Epoch: [{} / {}] | Step: {} / {} | Time: {:.2f}s | Loss: {:.4f} |"
                        " Val scores:".format(
                            epoch + 1,
                            self.config["training_args"]["epochs"],
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
                self.scheduler.step(val_scores[self.config["training_args"]["early_stop_metric"]])
            format_str = "Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} | Val scores:".format(
                epoch + 1, self.config["training_args"]["epochs"], np.mean(dur), np.mean(train_loss)
            )
            format_str += self.metric_to_str(val_scores)
            print(format_str)
            self.logger.write(format_str)

            if epoch > 0 and self.stopper.step(
                val_scores[self.config["training_args"]["early_stop_metric"]], self.model
            ):
                break

        return self.stopper.best_score

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for data in dataloader:
                data = all_to_cuda(data, self.config["env_args"]["device"])
                data["graph_data"] = data["graph_data"].to(self.config["env_args"]["device"])

                if self.use_copy:
                    oov_dict = prepare_ext_vocab(
                        data["graph_data"], self.vocab, device=self.config["env_args"]["device"]
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
                data = all_to_cuda(data, self.config["env_args"]["device"])
                data["graph_data"] = data["graph_data"].to(self.config["env_args"]["device"])
                if self.use_copy:
                    oov_dict = prepare_ext_vocab(
                        data["graph_data"], self.vocab, device=self.config["env_args"]["device"]
                    )
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                batch_gd = self.model.encode_init_node_feature(data)
                prob = self.model.g2s.encoder_decoder_beam_search(
                    batch_gd, self.config["inference_args"]["beam_size"], topk=1, oov_dict=oov_dict
                )

                pred_ids = (
                    torch.zeros(
                        len(prob),
                        self.config["model_args"]["decoder_args"]["rnn_decoder_private"][
                            "max_decoder_step"
                        ],
                    )
                    .fill_(ref_dict.EOS)
                    .to(self.config["env_args"]["device"])
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
            os.path.join(self.config["checkpoint_args"]["out_dir"], Constants._SAVED_WEIGHTS_FILE)
        ).to(self.config["env_args"]["device"])

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
    np.random.seed(config["env_args"]["seed"])
    torch.manual_seed(config["env_args"]["seed"])

    if not config["env_args"]["no_cuda"] and torch.cuda.is_available():
        print("[ Using CUDA ]")
        config["env_args"]["device"] = torch.device(
            "cuda" if config["env_args"]["gpu"] < 0 else "cuda:%d" % config["env_args"]["gpu"]
        )
        cudnn.benchmark = True
        torch.cuda.manual_seed(config["env_args"]["seed"])
    else:
        config["env_args"]["device"] = torch.device("cpu")

    print("\n" + config["checkpoint_args"]["out_dir"])

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
        print("{} -->  {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")

    cfg = get_args()
    config = load_json_config(cfg["json_config"])
    print_config(config)

    main(config)
