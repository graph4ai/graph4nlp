from copy import deepcopy
import stanfordcorenlp
import torch

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.data.dataset import SequenceLabelingDataset

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
        topology_subdir=None,
        pretrained_word_emb_cache_dir=None,
        edge_strategy=None,
        merge_strategy=None,
        tag_types=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        for_inference=None,
        reused_vocab_model=None,
        **kwargs
    ):
        super(ConllDataset, self).__init__(
            root_dir=root_dir,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            graph_name=graph_name,
            static_or_dynamic=static_or_dynamic,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            tag_types=tag_types,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
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
        graph_name,
        dynamic_init_topology_builder,
        merge_strategy,
        edge_strategy,
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
            self.lower_case,
            self.tokenizer,
            9000,
            1000,
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
        graph_name,
        static_or_dynamic="static",
        topology_builder=None,
        topology_subdir=None,
        pretrained_word_emb_cache_dir=None,
        edge_strategy=None,
        merge_strategy=None,
        tag_types=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        for_inference=None,
        reused_vocab_model=None,
        **kwargs
    ):
        super(ConllDataset_inference, self).__init__(
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            graph_name=graph_name,
            static_or_dynamic=static_or_dynamic,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            tag_types=tag_types,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
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
