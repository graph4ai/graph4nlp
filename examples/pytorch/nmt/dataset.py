from graph4nlp.pytorch.data.dataset import Text2TextDataItem, Text2TextDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
import torch
import os
import json
from stanfordcorenlp import StanfordCoreNLP


class EuroparlNMTDataset(Text2TextDataset):
    def __init__(self, root_dir, topology_builder, topology_subdir=None, graph_type='static',
                 edge_strategy=None, merge_strategy='tailhead', **kwargs):
        super(EuroparlNMTDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder, share_vocab=False,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy, **kwargs)

    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'train.json', 'test': 'test.json'}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    def download(self):
        raise NotImplementedError("It shouldn't be called now")


    def parse_file(self, files):
        with open(files, "r") as f:
            datalist = json.load(f)
        data = []
        for item in datalist:
            if item[0].strip() == "" or item[1].strip() == "":
                continue
            dataitem = Text2TextDataItem(input_text=item[0], output_text=item[1], tokenizer=self.tokenizer, share_vocab=self.share_vocab)
            data.append(dataitem)
        if len(data) > 10000:
            data = data[0:10000]
        return data

    def build_topology(self, data_items):
        """
        Build graph topology for each item in the dataset. The generated graph is bound to the `graph` attribute of the
        DataItem.
        """
        if self.graph_type == 'static':
            print('Connecting to stanfordcorenlp server...')
            processor = StanfordCoreNLP('http://localhost', port=9000, timeout=1000)
            print('CoreNLP server connected.')
            processor_args = {
                'annotators': 'ssplit,tokenize,depparse',
                "tokenize.options":
                    "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                "tokenize.whitespace": False,
                'ssplit.isOneSentence': False,
                'outputFormat': 'json'
            }
            cnt = 0
            for item in data_items:
                if cnt % 100 == 0:
                    print(cnt, len(data_items))
                cnt += 1
                graph = self.topology_builder.topology(raw_text_data=item.input_text,
                                                       nlp_processor=processor, processor_args=processor_args,
                                                       merge_strategy=self.merge_strategy,
                                                       edge_strategy=self.edge_strategy,
                                                       verbase=False)
                item.graph = graph
        elif self.graph_type == 'dynamic':
            for item in data_items:
                graph = self.topology_builder.raw_text_to_init_graph(item.input_text)
                item.graph = graph
        else:
            raise NotImplementedError('Currently only static and dynamic are supported!')


if __name__ == "__main__":
    dataset = EuroparlNMTDataset(root_dir="/home/shiina/shiina/lib/dataset",
                                 topology_builder=DependencyBasedGraphConstruction,
                                 topology_subdir='DependencyGraph')


