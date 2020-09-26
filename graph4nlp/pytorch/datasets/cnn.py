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
        Add exception process for IE-based graph construction. When the input text is too long,
        the processor may raise ``java.util.concurrent.TimeoutException``
        """
        if self.graph_type == 'static':
            print('Connecting to stanfordcorenlp server...')
            processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=9098, timeout=1000)

            if self.topology_builder == IEBasedGraphConstruction:
                props_coref = {
                    'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref',
                    "tokenize.options":
                        "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                    "tokenize.whitespace": False,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json'
                }
                props_openie = {
                    'annotators': 'tokenize, ssplit, pos, ner, parse, openie',
                    "tokenize.options":
                        "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                    "tokenize.whitespace": False,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json',
                    "openie.triple.strict": "true"
                }
                processor_args = [props_coref, props_openie]
            elif self.topology_builder == DependencyBasedGraphConstruction:
                processor_args = {
                    'annotators': 'ssplit,tokenize,depparse',
                    "tokenize.options":
                        "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                    "tokenize.whitespace": False,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json'
                }
            elif self.topology_builder == ConstituencyBasedGraphConstruction:
                processor_args = {
                    'annotators': "tokenize,ssplit,pos,parse",
                    "tokenize.options":
                    "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                    "tokenize.whitespace": False,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json'
                }
            else:
                raise NotImplementedError
            print('CoreNLP server connected.')
            pop_idxs = []
            for idx, item in enumerate(data_items):
                try:
                    graph = self.topology_builder.topology(raw_text_data=item.input_text,
                                                           nlp_processor=processor,
                                                           processor_args=processor_args,
                                                           merge_strategy=self.merge_strategy,
                                                           edge_strategy=self.edge_strategy,
                                                           verbase=False)
                    item.graph = graph
                except:
                    pop_idxs.append(idx)
                    item.graph = None
                    print('item does not have graph: '+str(idx))

            data_items = [x for idx, x in enumerate(data_items) if idx not in pop_idxs]
        elif self.graph_type == 'dynamic':
            if self.dynamic_graph_type == 'node_emb':
                for item in data_items:
                    graph = self.topology_builder.init_topology(item.input_text,
                                                                lower_case=self.lower_case,
                                                                tokenizer=self.tokenizer)
                    item.graph = graph
            elif self.dynamic_graph_type == 'node_emb_refined':
                if self.dynamic_init_topology_builder in (IEBasedGraphConstruction, DependencyBasedGraphConstruction, ConstituencyBasedGraphConstruction):
                    print('Connecting to stanfordcorenlp server...')
                    processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=9000, timeout=1000)

                    if self.dynamic_init_topology_builder == IEBasedGraphConstruction:
                        props_coref = {
                            'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref',
                            "tokenize.options":
                                "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json'
                        }
                        props_openie = {
                            'annotators': 'tokenize, ssplit, pos, ner, parse, openie',
                            "tokenize.options":
                                "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json',
                            "openie.triple.strict": "true"
                        }
                        processor_args = [props_coref, props_openie]
                    elif self.dynamic_init_topology_builder == DependencyBasedGraphConstruction:
                        processor_args = {
                            'annotators': 'ssplit,tokenize,depparse',
                            "tokenize.options":
                                "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json'
                        }
                    elif self.dynamic_init_topology_builder == ConstituencyBasedGraphConstruction:
                        processor_args = {
                            'annotators': "tokenize,ssplit,pos,parse",
                            "tokenize.options":
                            "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json'
                        }
                    else:
                        raise NotImplementedError
                    print('CoreNLP server connected.')
                else:
                    processor = None
                    processor_args = None

                for item in data_items:
                    graph = self.topology_builder.init_topology(item.input_text,
                                                                dynamic_init_topology_builder=self.dynamic_init_topology_builder,
                                                                lower_case=self.lower_case,
                                                                tokenizer=self.tokenizer,
                                                                nlp_processor=processor,
                                                                processor_args=processor_args,
                                                                merge_strategy=self.merge_strategy,
                                                                edge_strategy=self.edge_strategy,
                                                                verbase=False,
                                                                dynamic_init_topology_aux_args=self.dynamic_init_topology_aux_args)

                    item.graph = graph
            else:
                raise RuntimeError('Unknown dynamic_graph_type: {}'.format(self.dynamic_graph_type))

        else:
            raise NotImplementedError('Currently only static and dynamic are supported!')

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