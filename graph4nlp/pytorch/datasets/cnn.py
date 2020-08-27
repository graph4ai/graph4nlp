import os

import torch
import pickle
import json
from graph4nlp.pytorch.data.dataset import Text2TextDataset, Text2TextDataItem
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel
from ..modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from ..modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction
import stanfordcorenlp
import time
dataset_root = '../../../examples/pytorch/summarization/cnn'


class CNNDataset(Text2TextDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'train.json', 'test': 'test.json', 'val': 'val.json'}

    @property
    def processed_file_names(self):
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')
        return

    def __init__(self, root_dir, topology_builder=None, topology_subdir=None, graph_type='static',
                 edge_strategy=None, merge_strategy=None, **kwargs):
        super(CNNDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy, **kwargs)

    def build_topology(self, data_items):
        """
        Build graph topology for each item in the dataset. The generated graph is bound to the `graph` attribute of the
        DataItem.
        """
        if self.graph_type == 'static':
            print('Connecting to stanfordcorenlp server...')
            processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=9000, timeout=1000)
            print('CoreNLP server connected.')
            pop_idxs = []
            for idx, item in enumerate(data_items):
                try:
                    graph = self.topology_builder.topology(raw_text_data=item.input_text, nlp_processor=processor,
                                                           merge_strategy=self.merge_strategy,
                                                           edge_strategy=self.edge_strategy,
                                                           verbase=False)
                    item.graph = graph
                    if idx%100==1:
                        print(idx)
                        # torch.save(data_items, 'cnn_topo_tmp.pt')
                except:
                    pop_idxs.append(idx)
                    item.graph = None
                    print('item does not have graph: '+str(idx))

            data_items = [x for idx, x in enumerate(data_items) if idx not in pop_idxs]
            return data_items
        elif self.graph_type == 'dynamic':
            # TODO: Implement this
            pass
        else:
            raise NotImplementedError('Currently only static and dynamic are supported!')

    def _process(self):
        if all([os.path.exists(processed_path) for processed_path in self.processed_file_paths.values()]):
            if 'val_split_ratio' in self.__dict__:
                UserWarning(
                    "Loading existing processed files on disk. Your `val_split_ratio` might not work since the data have"
                    "already been split.")
            return

        os.makedirs(self.processed_dir, exist_ok=True)

        self.read_raw_data()

        self.train = self.build_topology(self.train)
        self.test = self.build_topology(self.test)
        if 'val' in self.__dict__:
            self.val = self.build_topology(self.val)

        self.build_vocab()

        self.vectorization(self.train)
        self.vectorization(self.test)
        if 'val' in self.__dict__:
            self.vectorization(self.val)

        data_to_save = {'train': self.train, 'test': self.test}
        if 'val' in self.__dict__:
            data_to_save['val'] = self.val
        torch.save(data_to_save, self.processed_file_paths['data'])

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format is specified by each individual task-specific
        base class. Returns all the indices of data items in this file w.r.t. the whole dataset.
        For Text2TextDataset, the format of the input file should contain lines of input, each line representing one
        record of data. The input and output is separated by a tab(\t).
        Examples
        --------
        input: list job use languageid0 job ( ANS ) , language ( ANS , languageid0 )
        DataItem: input_text="list job use languageid0", output_text="job ( ANS ) , language ( ANS , languageid0 )"
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
        with open(file_path, 'r') as f:
            examples = json.load(f)
            for example_dict in examples:
                input = ' '.join(example_dict['article'][:10])
                output = example_dict['highlight'][0][0]
                data_item = Text2TextDataItem(input_text=input, output_text=output, tokenizer=self.tokenizer,
                                              share_vocab=self.share_vocab)
                data.append(data_item)
        return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--parser', type=str, default='IE')
    args = parser.parse_args()

    if args.parser=='IE':
        start_time = time.time()
        cnn_ie = CNNDataset(root_dir=dataset_root, topology_builder=IEBasedGraphConstruction,
                    topology_subdir='IEGraph')
        end_time = time.time() # 333.8594479560852
        # 141.86208820343018 [:10]
    elif args.parser=='DEP':
        start_time = time.time()
        cnn_dep = CNNDataset(root_dir=dataset_root, topology_builder=DependencyBasedGraphConstruction,
                        topology_subdir='DepGraph')
        end_time = time.time() # 22.60792326927185
    elif args.parser=='CONS':
        start_time = time.time()
        cnn_cons = CNNDataset(root_dir=dataset_root, topology_builder=ConstituencyBasedGraphConstruction,
                             topology_subdir='ConsGraph')
        end_time = time.time() # 48.007439851760864
    else:
        raise NotImplementedError()

    print(end_time - start_time)

    a = 0