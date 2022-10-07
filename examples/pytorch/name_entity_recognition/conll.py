import torch

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.data.dataset import SequenceLabelingDataset

from copy import deepcopy


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
