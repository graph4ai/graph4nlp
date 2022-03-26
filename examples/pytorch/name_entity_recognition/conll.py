import os
from copy import deepcopy
import stanfordcorenlp
import stanza
import torch
from stanza.server import CoreNLPClient

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.data.dataset import SequenceLabelingDataset
from graph4nlp.pytorch.modules.utils.nlp_parser_utils import get_stanza_properties

from dependency_graph_construction_without_tokenize import (
    DependencyBasedGraphConstruction_without_tokenizer,
)
from line_graph_construction import LineBasedGraphConstruction


class ConllDataset(SequenceLabelingDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {"train": "eng.train", "test": "eng.testa", "val": "eng.testb"}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        # return {'vocab': 'vocab.pt', 'data': 'data.pt', 'split_ids': 'split_ids.pt'}
        return {"vocab": "vocab.pt", "data": "data.pt"}

    # def download(self):
    #    raise NotImplementedError(
    #        'This dataset is now under test and cannot be downloaded.
    # Please prepare the raw data yourself.')

    def __init__(
        self,
        root_dir,
        graph_name,
        static_or_dynamic="static",
        topology_builder=None,
        pretrained_word_emb_cache_dir=None,
        edge_strategy=None,
        merge_strategy=None,
        tag_types=None,
        topology_subdir=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
        for_inference=None,
        nlp_processor_args=None,
        reused_vocab_model=None,
        **kwargs
    ):
        super(ConllDataset, self).__init__(
            root_dir=root_dir,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            graph_construction_name=graph_name,
            static_or_dynamic=static_or_dynamic,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            tag_types=tag_types,
            nlp_processor_args=nlp_processor_args,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
            **kwargs
        )

        self.dynamic_init_topology_builder = dynamic_init_topology_builder

    def _build_topology_process(
        self,
        data_items,
        topology_builder,
        nlp_processor,
        nlp_processor_args,
        static_or_dynamic,
        graph_name,
        dynamic_init_topology_builder,
        merge_strategy,
        edge_strategy,
        dynamic_init_topology_aux_args,
        lower_case,
        tokenizer,
        port,
        timeout,
    ):
        """
        Build graph topology for each item in the dataset. The generated graph is bound to
        the `graph` attribute of the DataItem.
        """

        if self.static_or_dynamic not in ["static", "dynamic"]:
            raise ValueError("Argument: ``static_or_dynamic`` must be ``static`` or ``dynamic``")

        if self.static_or_dynamic == "static":

            if self.topology_builder == DependencyBasedGraphConstruction_without_tokenizer:
                nlp_processor.start()
                for item in data_items:
                    graph = self.topology_builder.static_topology(
                        raw_text_data=item.input_text,
                        nlp_processor=nlp_processor,
                        processor_args=nlp_processor_args,
                        merge_strategy=merge_strategy,
                        edge_strategy=edge_strategy,
                        verbose=False,
                        auxiliary_args=self.depedency_topology_aux_args,
                    )
                    item.graph = graph
                ret = [x for idx, x in enumerate(data_items)]

            elif self.topology_builder == LineBasedGraphConstruction:
                processor = None
                for item in data_items:
                    graph = self.topology_builder.static_topology(
                        raw_text_data=item.input_text,
                        merge_strategy=self.merge_strategy,
                        edge_strategy=self.edge_strategy,
                        nlp_processor=None,
                        processor_args=None,
                    )
                    item.graph = graph
                ret = [x for idx, x in enumerate(data_items)]

            else:
                raise NotImplementedError("unknown static graph type: {}".format(self.graph_name))

            print("CoreNLP server is connected!")

        elif self.static_or_dynamic == "dynamic":
            if self.graph_name == "node_emb":
                for item in data_items:
                    graph = self.topology_builder.init_topology(
                        item.input_text, lower_case=self.lower_case, tokenizer=self.tokenizer
                    )

                    item.graph = graph

            elif self.graph_name == "node_emb_refined":

                if (
                    self.dynamic_init_topology_builder
                    == DependencyBasedGraphConstruction_without_tokenizer
                ):
                    print("Connecting to stanfordcorenlp server...")
                    processor = stanfordcorenlp.StanfordCoreNLP(
                        "http://localhost", port=9000, timeout=1000
                    )
                    print("CoreNLP server connected.")
                    processor_args = {
                        "annotators": "ssplit,tokenize,depparse",
                        "tokenize.options": "splitHyphenated=false,normalizeParentheses=false,\
                            normalizeOtherBrackets=false",
                        "tokenize.whitespace": False,
                        "ssplit.isOneSentence": False,
                        "outputFormat": "json",
                    }
                else:
                    processor = None
                    processor_args = None

                self.dynamic_init_topology_aux_args = {
                    "lower_case": self.lower_case,
                    "tokenizer": self.tokenizer,
                    "merge_strategy": self.merge_strategy,
                    "edge_strategy": self.edge_strategy,
                    "verbose": False,
                    "nlp_processor": processor,
                    "processor_args": processor_args,
                }

                for item in data_items:
                    graph = self.topology_builder.init_topology(
                        item.input_text,
                        dynamic_init_topology_builder=self.dynamic_init_topology_builder,
                        dynamic_init_topology_aux_args=self.dynamic_init_topology_aux_args,
                    )

                    item.graph = graph
            else:
                raise RuntimeError("Unknown graph_name: {}".format(self.dynamic_graph_name))

        else:
            raise NotImplementedError("Currently only static and dynamic are supported!")
        return ret

    def build_topology(self, data_items):
        print("building topology")
        if self.topology_builder == LineBasedGraphConstruction:
            nlp_processor = None
            nlp_processor_args = None
        else:
            if self.nlp_processor_args["name"] == "stanza":
                corenlp_dir = self.nlp_processor_args["args"]["corenlp_dir"]
                stanza.install_corenlp(dir=corenlp_dir)
                os.environ["CORENLP_HOME"] = corenlp_dir
                print("Connecting to NLP parsing tool: stanza")
                from stanza.server.client import StartServer

                nlp_processor = CoreNLPClient(
                    annotators=self.nlp_processor_args["args"]["annotators"],
                    start_server=StartServer.TRY_START,
                    memory=self.nlp_processor_args["args"]["memory"],
                    endpoint=self.nlp_processor_args["args"]["endpoint"],
                    be_quiet=True,
                    output_format="json",
                )

                nlp_processor_args = get_stanza_properties(
                    self.nlp_processor_args["args"]["properties"]
                )

        return self._build_topology_process(
            data_items,
            self.topology_builder,
            nlp_processor,
            nlp_processor_args,
            self.static_or_dynamic,
            self.graph_construction_name,
            self.dynamic_init_topology_builder,
            self.merge_strategy,
            self.edge_strategy,
            self.dynamic_init_topology_aux_args,
            self.lower_case,
            self.tokenizer,
            self.port,
            self.timeout,
        )


class ConllDataset_inference(SequenceLabelingDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {"train": "eng.train", "test": "eng.testa", "val": "eng.testb"}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        # return {'vocab': 'vocab.pt', 'data': 'data.pt', 'split_ids': 'split_ids.pt'}
        return {"vocab": "vocab.pt", "data": "data.pt"}

    # def download(self):
    #    raise NotImplementedError(
    #        'This dataset is now under test and cannot be downloaded.
    # Please prepare the raw data yourself.')

    def __init__(
        self,
        graph_construction_name,
        static_or_dynamic="static",
        topology_builder=None,
        topology_subdir=None,
        pretrained_word_emb_cache_dir=None,
        edge_strategy=None,
        merge_strategy=None,
        tag_types=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
        for_inference=None,
        nlp_processor_args=None,
        reused_vocab_model=None,
        **kwargs
    ):
        super(ConllDataset_inference, self).__init__(
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            graph_construction_name=graph_construction_name,
            static_or_dynamic=static_or_dynamic,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            tag_types=tag_types,
            nlp_processor_args=nlp_processor_args,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
            **kwargs
        )

        self.dynamic_init_topology_builder = dynamic_init_topology_builder

    def _build_topology_process(
        self,
        data_items,
        topology_builder,
        static_or_dynamic,
        graph_construction_name,
        dynamic_init_topology_builder,
        merge_strategy,
        edge_strategy,
        dynamic_init_topology_aux_args,
        lower_case,
        tokenizer,
        port,
        timeout,
    ):
        """
        Build graph topology for each item in the dataset. The generated graph is bound to
        the `graph` attribute of the DataItem.
        """

        print("using conll dataset!")
        if self.static_or_dynamic not in ["static", "dynamic"]:
            raise ValueError("Argument: ``static_or_dynamic`` must be ``static`` or ``dynamic``")

        if self.static_or_dynamic == "static":

            if self.topology_builder == DependencyBasedGraphConstruction_without_tokenizer:
                print("Connecting to stanfordcorenlp server...")
                processor = stanfordcorenlp.StanfordCoreNLP(
                    "http://localhost", port=9000, timeout=1000
                )
                processor_args = {
                    "annotators": "ssplit,tokenize,depparse",
                    "tokenize.options": "splitHyphenated=false,normalizeParentheses=false,\
                        normalizeOtherBrackets=false",
                    "tokenize.whitespace": False,
                    "ssplit.isOneSentence": False,
                    "outputFormat": "json",
                }
                for item in data_items:
                    graph = self.topology_builder.static_topology(
                        raw_text_data=item.input_text,
                        auxiliary_args=self.depedency_topology_aux_args,
                    )
                    item.graph = graph

            elif self.topology_builder == LineBasedGraphConstruction:
                processor = None
                processor_args = {
                    "annotators": "ssplit,tokenize,depparse",
                    "tokenize.options": "splitHyphenated=false,normalizeParentheses=false,\
                        normalizeOtherBrackets=false",
                    "tokenize.whitespace": False,
                    "ssplit.isOneSentence": False,
                    "outputFormat": "json",
                }
                for item in data_items:
                    graph = self.topology_builder.static_topology(
                        raw_text_data=item.input_text,
                        merge_strategy=self.merge_strategy,
                        edge_strategy=self.edge_strategy,
                        nlp_processor=None,
                        processor_args=None,
                    )
                    item.graph = graph

            else:
                raise NotImplementedError("unknown static graph type: {}".format(self.graph_name))

            print("CoreNLP server is connected!")

        elif self.static_or_dynamic == "dynamic":
            if self.graph_name == "node_emb":
                for item in data_items:
                    graph = self.topology_builder.init_topology(
                        item.input_text, lower_case=self.lower_case, tokenizer=self.tokenizer
                    )

                    item.graph = graph

            elif self.graph_name == "node_emb_refined":

                if (
                    self.dynamic_init_topology_builder
                    == DependencyBasedGraphConstruction_without_tokenizer
                ):
                    print("Connecting to stanfordcorenlp server...")
                    processor = stanfordcorenlp.StanfordCoreNLP(
                        "http://localhost", port=9000, timeout=1000
                    )
                    print("CoreNLP server connected.")
                    processor_args = {
                        "annotators": "ssplit,tokenize,depparse",
                        "tokenize.options": "splitHyphenated=false,normalizeParentheses=false,\
                            normalizeOtherBrackets=false",
                        "tokenize.whitespace": False,
                        "ssplit.isOneSentence": False,
                        "outputFormat": "json",
                    }
                else:
                    processor = None
                    processor_args = None

                self.dynamic_init_topology_aux_args = {
                    "lower_case": self.lower_case,
                    "tokenizer": self.tokenizer,
                    "merge_strategy": self.merge_strategy,
                    "edge_strategy": self.edge_strategy,
                    "verbose": False,
                    "nlp_processor": processor,
                    "processor_args": processor_args,
                }

                for item in data_items:
                    graph = self.topology_builder.init_topology(
                        item.input_text,
                        dynamic_init_topology_builder=self.dynamic_init_topology_builder,
                        dynamic_init_topology_aux_args=self.dynamic_init_topology_aux_args,
                    )

                    item.graph = graph
            else:
                raise RuntimeError("Unknown graph_name: {}".format(self.dynamic_graph_name))

        else:
            raise NotImplementedError("Currently only static and dynamic are supported!")
        return data_items

    def build_topology(self, data_items):
        return self._build_topology_process(
            data_items,
            self.topology_builder,
            self.static_or_dynamic,
            self.graph_name,
            self.dynamic_init_topology_builder,
            self.merge_strategy,
            self.edge_strategy,
            self.dynamic_init_topology_aux_args,
            self.lower_case,
            self.tokenizer,
            9000,
            1000,
        )

    def _vectorize_one_dataitem(self, data_item, vocab_model, use_ie=None):
        item = deepcopy(data_item)
        graph: GraphData = item.graph
        token_matrix = []
        for node_idx in range(graph.get_node_num()):
            node_token = graph.node_attributes[node_idx]["token"]
            node_token_id = vocab_model.in_word_vocab.getIndex(node_token)
            graph.node_attributes[node_idx]["token_id"] = node_token_id
            token_matrix.append([node_token_id])
        token_matrix = torch.tensor(token_matrix, dtype=torch.long)
        graph.node_features["token_id"] = token_matrix

        item.output_id = None
        return item

    def vectorization(self, data_items):
        for idx in range(len(data_items)):
            data_items[idx] = self._vectorize_one_dataitem(data_items[idx])
