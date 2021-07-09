import os

import torch
import torch.utils.data
from nltk.tokenize import word_tokenize

from graph4nlp.pytorch.data.data import GraphData
from graph4nlp.pytorch.data.data import to_batch
from graph4nlp.pytorch.data.dataset import Dataset, DataItem
from graph4nlp.pytorch.modules.graph_construction import DependencyBasedGraphConstruction


class SequenceDataItem(DataItem):
    def __init__(self, input_text, tokenizer):
        super(SequenceDataItem, self).__init__(input_text, tokenizer)

    def extract(self):
        g: GraphData = self.graph

        input_tokens = []
        for i in range(g.get_node_num()):
            tokenized_token = self.tokenizer(g.node_attributes[i]['token'])
            input_tokens.extend(tokenized_token)

        return input_tokens


class SentencesDataset(Dataset):
    """
    Class for sentences dataset.
    Converts them to graphs, puts them in dataset, so as to be used in inference
    Have to derive from Dataset so that DataLoader can accept it.
    """
    @property
    def raw_file_names(self): # Not used
        return {'train':'eng.train','test':'eng.testa','val':'eng.testb'}

    @property
    def processed_dir(self) -> str: # Used to get pre-cooked vocab.pt
        return os.path.join(self.root, 'processed', self.topology_subdir)

    @property
    def processed_file_paths(self) -> dict: # Used to get pre-cooked vocab.pt
        return {name: os.path.join(self.processed_dir, processed_file_name) for name, processed_file_name in
                self.processed_file_names.items()}

    @property
    def processed_file_names(self): # Used to get pre-cooked vocab.pt
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    def download(self):
        print("Not Implemented")

    def parse_file(self, file_path) -> list:
        print("Not Implemented")
        return []

    def __init__(self,sentences,
                 root_dir="",
                 topology_builder=DependencyBasedGraphConstruction,
                 topology_subdir='DependencyGraph',
                 tokenizer=word_tokenize,
                 # lower_case=True,
                 graph_type='static',
                 pretrained_word_emb_name="6B",
                 # pretrained_word_emb_url=None,
                 # target_pretrained_word_emb_name=None,
                 # target_pretrained_word_emb_url=None,
                 pretrained_word_emb_cache_dir=".vector_cache/",
                 edge_strategy=None,
                 merge_strategy=None,
                 tag_types=None,
                 dynamic_init_graph_type=None,
                 dynamic_graph_type=None,
                 dynamic_init_topology_builder=None,
                 dynamic_init_topology_aux_args=None,
                 # max_word_vocab_size=None,
                 # min_word_vocab_freq=1,
                 # use_val_for_vocab=False,
                 # seed=1234,
                 # thread_number=4,
                 # port=9000,
                 # timeout=15000,
                 word_emb_size=300,
                 share_vocab=True,
                 **kwargs):
        super(SentencesDataset, self).__init__(
            root = root_dir, topology_builder = topology_builder,
            topology_subdir = topology_subdir, graph_type = graph_type,
            edge_strategy = edge_strategy, merge_strategy = merge_strategy, tag_types = tag_types,
            dynamic_init_topology_builder = dynamic_init_topology_builder,
            pretrained_word_emb_name = pretrained_word_emb_name,
            share_vocab=share_vocab,word_emb_size=word_emb_size,
            pretrained_word_emb_cache_dir = pretrained_word_emb_cache_dir, ** kwargs)

        self.tokenizer = tokenizer
        self.sentences = sentences
        self.dynamic_init_graph_type = dynamic_init_graph_type
        self.dynamic_graph_type = dynamic_graph_type
        self.dynamic_init_topology_aux_args = dynamic_init_topology_aux_args

        self.process() # cannot name it _process as base class subfunctions get called, but TBD


    def vectorization(self,data_items):
        for item in data_items:
            graph: GraphData = item.graph
            token_matrix = []
            for node_idx in range(graph.get_node_num()):
                node_token = graph.node_attributes[node_idx]['token']
                node_token_id = self.vocab_model.in_word_vocab.getIndex(node_token)
                graph.node_attributes[node_idx]['token_id'] = node_token_id
                token_matrix.append([node_token_id])
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features['token_id'] = token_matrix

            # NO OUTPUT TAGS
            # tgt = item.output_tag
            # tgt_tag_id = [self.tag_types.index(tgt_.strip()) for tgt_ in tgt]
            #
            # tgt_tag_id = torch.tensor(tgt_tag_id)
            # item.output_id = tgt_tag_id

    @staticmethod
    def collate_fn(data_list: [SequenceDataItem]):

        graph_list = [item.graph for item in data_list]
        graph_data = to_batch(graph_list)
        # tgt_tag = [deepcopy(item.output_id) for item in data_list]

        return {"graph_data": graph_data}
                # "tgt_tag": tgt_tag}

    def read_raw_data(self):
        data_items = []
        for line in self.sentences:
            words = self.tokenizer(line)
            for word in words:
                data_item = SequenceDataItem(input_text=word, tokenizer=self.tokenizer)
                data_items.append(data_item)
        return data_items

    # NOT SURE WHY THESE HAVE TO BE IMPLEMENTED UNLIKE SentenceLabeledDataset
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def process(self):
        self.data_list = self.read_raw_data()
        self.data_list = self.build_topology(self.data_list)
        # # self.build_vocab() # Dataset:build vocab is good enough as just need to load already created vocab
        self.vectorization(self.data_list)
