from graph4nlp.pytorch.data.dataset import Text2TextDataItem, Text2TextDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
import torch
import os
import json
from stanfordcorenlp import StanfordCoreNLP
from multiprocessing import Pool
import numpy as np

from multiprocessing import Process
import multiprocessing
import tqdm


class EuroparlNMTDataset(Text2TextDataset):
    def __init__(self, root_dir, topology_builder, topology_subdir=None, graph_type='static',
                 edge_strategy=None, merge_strategy='tailhead', share_vocab=False):
        super(EuroparlNMTDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder, share_vocab=share_vocab,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy)

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
        max_len = -1
        for item in datalist:
            if item[0].strip() == "" or item[1].strip() == "":
                continue
            input_tokens = item[0].split(" ")
            if len(input_tokens) >= 50:
                continue
            max_len = max(max_len, len(input_tokens))
            dataitem = Text2TextDataItem(input_text=item[0], output_text=item[1], tokenizer=self.tokenizer, share_vocab=self.share_vocab)
            data.append(dataitem)
        if len(data) > 200000:
            data = data[:200000]
        return data

    @staticmethod
    def process(data_item, port):
        processor_args = {
            'annotators': 'ssplit,tokenize,depparse',
            "tokenize.options":
                "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
            "tokenize.whitespace": False,
            'ssplit.isOneSentence': False,
            'outputFormat': 'json'
        }
        print('Connecting to stanfordcorenlp server...')
        processor = StanfordCoreNLP('http://localhost', port=int(port), timeout=1000)
        processor.switch_language("fr")
        print('CoreNLP server connected.')
        cnt = 0
        all = len(data_item)
        ret = []
        # print(id(data_item[0]), data_item[0].input_text)
        for item in data_item:
            if cnt % 1000 == 0:
                print("Port {}, processing: {} / {}".format(port, cnt, all))
            cnt += 1
            # print(item.input_text)
            graph = DependencyBasedGraphConstruction.topology(raw_text_data=item.input_text,
                                                               nlp_processor=processor, processor_args=processor_args,
                                                               merge_strategy="tailhead",
                                                               edge_strategy=None,
                                                               verbase=False)
            ret.append(graph)
        print("Port {}, finish".format(port))
        return ret



    def build_topology(self, data_items):
        """
        Build graph topology for each item in the dataset. The generated graph is bound to the `graph` attribute of the
        DataItem.
        """
        # print(id(data_items[0]), data_items[0].input_text)

        total = len(data_items)
        pool = Pool(10)
        res_l = []
        for i in range(10):
            start_index = total * i // 10
            end_index = total * (i + 1) // 10
            r = pool.apply_async(self.process, args=(data_items[start_index:end_index], 9000+i))
            res_l.append(r)
        pool.close()
        pool.join()

        for i in range(10):
            start_index = total * i // 10
            end_index = total * (i + 1) // 10

            res = res_l[i].get()
            datas = data_items[start_index:end_index]
            for data, graph in zip(datas, res):
                data.graph = graph


if __name__ == "__main__":
    dataset = EuroparlNMTDataset(root_dir="/home/shiina/shiina/lib/dataset",
                                 topology_builder=DependencyBasedGraphConstruction,
                                 topology_subdir='DependencyGraph20', share_vocab=False)


