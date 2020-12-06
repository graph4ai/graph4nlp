from graph4nlp.pytorch.data.dataset import Text2TextDataItem, Text2TextDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.constituency_graph_construction import ConstituencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
import torch
import os
import json
import stanfordcorenlp
import warnings
from multiprocessing import Pool
import numpy as np
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size

from multiprocessing import Process
import multiprocessing
import tqdm
from graph4nlp.pytorch.modules.utils.vocab_utils import VocabModel, Vocab
from graph4nlp.pytorch.modules.utils import constants
from nltk.tokenize import word_tokenize

class CNNDataset(Text2TextDataset):
    def __init__(self,
                 root_dir,
                 topology_builder,
                 topology_subdir,
                 tokenizer=word_tokenize,
                 lower_case=True,
                 pretrained_word_emb_file=None,
                 use_val_for_vocab=False,
                 seed=1234,
                 device='cpu',
                 thread_number=4,
                 port=9000,
                 timeout=15000,
                 graph_type='static',
                 edge_strategy=None,
                 merge_strategy='tailhead',
                 share_vocab=True,
                 word_emb_size=300,
                 dynamic_graph_type=None,
                 dynamic_init_topology_builder=None,
                 dynamic_init_topology_aux_args=None
                 ):
        super(CNNDataset, self).__init__(root_dir=root_dir,
                                         topology_builder=topology_builder,
                                         topology_subdir=topology_subdir,
                                         tokenizer=tokenizer,
                                         lower_case=lower_case,
                                         pretrained_word_emb_file=pretrained_word_emb_file,
                                         use_val_for_vocab=use_val_for_vocab,
                                         seed=seed,
                                         device=device,
                                         thread_number=thread_number,
                                         port=port,
                                         timeout=timeout,
                                         graph_type=graph_type,
                                         edge_strategy=edge_strategy,
                                         merge_strategy=merge_strategy,
                                         share_vocab=share_vocab,
                                         word_emb_size=word_emb_size,
                                         dynamic_graph_type=dynamic_graph_type,
                                         dynamic_init_topology_builder=dynamic_init_topology_builder,
                                         dynamic_init_topology_aux_args=dynamic_init_topology_aux_args)

    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        # return {'train': 'train_300.json', 'val': "train_30.json", 'test': 'train_30.json'}
        # return {'train': 'train_300.json', 'val': "train_300.json", 'test': 'train_300.json'}
        # return {'train': 'train_3k.json', 'val': "val.json", 'test': 'test.json'}
        return {'train': 'train_1w.json', 'val': "val.json", 'test': 'test.json'}
        # return {'train': 'train_3w.json', 'val': "val.json", 'test': 'test.json'}
        # return {'train': 'train_9w.json', 'val': "val.json", 'test': 'test.json'}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    @staticmethod
    def _build_topology_process(data_items, topology_builder,
                                graph_type, dynamic_graph_type, dynamic_init_topology_builder,
                                merge_strategy, edge_strategy, dynamic_init_topology_aux_args,
                                lower_case, tokenizer, port, timeout):

        ret = []
        if graph_type == 'static':
            print('Connecting to stanfordcorenlp server...')
            processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=port, timeout=timeout)

            if topology_builder == IEBasedGraphConstruction:
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
            elif topology_builder == DependencyBasedGraphConstruction:
                processor_args = {
                    'annotators': 'ssplit,tokenize,depparse',
                    "tokenize.options":
                        "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                    "tokenize.whitespace": False,
                    'ssplit.isOneSentence': False,
                    'outputFormat': 'json'
                }
            elif topology_builder == ConstituencyBasedGraphConstruction:
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
            # for cnt, item in enumerate(data_items):
            #     if cnt % 1000 == 0:
            #         print("Port {}, processing: {} / {}".format(port, cnt, len(data_items)))
            #     try:
            #         graph = topology_builder.topology(raw_text_data=item.input_text,
            #                                           nlp_processor=processor,
            #                                           processor_args=processor_args,
            #                                           merge_strategy=merge_strategy,
            #                                           edge_strategy=edge_strategy,
            #                                           verbase=False)
            #         ret.append(graph)
            #     except Exception as msg:
            #         warnings.warn(RuntimeWarning(msg))
            #         data_items.pop(data_items.index(item))

            pop_idxs = []
            for cnt, item in enumerate(data_items):
                if cnt % 1000 == 0:
                    print("Port {}, processing: {} / {}".format(port, cnt, len(data_items)))
                try:
                    graph = topology_builder.topology(raw_text_data=item.input_text,
                                                      nlp_processor=processor,
                                                      processor_args=processor_args,
                                                      merge_strategy=merge_strategy,
                                                      edge_strategy=edge_strategy,
                                                      verbase=False)
                    item.graph = graph
                    ret.append(item)
                except Exception as msg:
                    pop_idxs.append(cnt)
                    item.graph = None
                    warnings.warn(RuntimeWarning(msg))
            ret = [x for idx, x in enumerate(ret) if idx not in pop_idxs]

        elif graph_type == 'dynamic':
            if dynamic_graph_type == 'node_emb':
                for item in data_items:
                    graph = topology_builder.init_topology(item.input_text,
                                                           lower_case=lower_case,
                                                           tokenizer=tokenizer)
                    ret.append(graph)
            elif dynamic_graph_type == 'node_emb_refined':
                if dynamic_init_topology_builder in (
                        IEBasedGraphConstruction, DependencyBasedGraphConstruction, ConstituencyBasedGraphConstruction):
                    print('Connecting to stanfordcorenlp server...')
                    processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=port, timeout=timeout)

                    if dynamic_init_topology_builder == IEBasedGraphConstruction:
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
                    elif dynamic_init_topology_builder == DependencyBasedGraphConstruction:
                        processor_args = {
                            'annotators': 'ssplit,tokenize,depparse',
                            "tokenize.options":
                                "splitHyphenated=false,normalizeParentheses=false,normalizeOtherBrackets=false",
                            "tokenize.whitespace": False,
                            'ssplit.isOneSentence': False,
                            'outputFormat': 'json'
                        }
                    elif dynamic_init_topology_builder == ConstituencyBasedGraphConstruction:
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
                pop_idxs = []
                for idx, item in enumerate(data_items):
                    try:
                        graph = topology_builder.init_topology(item.input_text,
                                                               dynamic_init_topology_builder=dynamic_init_topology_builder,
                                                               lower_case=lower_case,
                                                               tokenizer=tokenizer,
                                                               nlp_processor=processor,
                                                               processor_args=processor_args,
                                                               merge_strategy=merge_strategy,
                                                               edge_strategy=edge_strategy,
                                                               verbase=False,
                                                               dynamic_init_topology_aux_args=dynamic_init_topology_aux_args)

                        ret.append(graph)
                    except Exception as msg:
                        pop_idxs.append(idx)
                        item.graph = None
                        warnings.warn(RuntimeWarning(msg))
                ret = [x for idx, x in enumerate(ret) if idx not in pop_idxs]
            else:
                raise RuntimeError('Unknown dynamic_graph_type: {}'.format(dynamic_graph_type))

        else:
            raise NotImplementedError('Currently only static and dynamic are supported!')

        return ret

    def build_topology(self, data_items):
        """
        Build graph topology for each item in the dataset. The generated graph is bound to the `graph` attribute of the
        DataItem.
        """
        total = len(data_items)
        thread_number = min(total, self.thread_number)
        pool = Pool(thread_number)
        res_l = []
        for i in range(thread_number):
            start_index = total * i // thread_number
            end_index = total * (i + 1) // thread_number

            """
            data_items, topology_builder,
                                graph_type, dynamic_graph_type, dynamic_init_topology_builder,
                                merge_strategy, edge_strategy, dynamic_init_topology_aux_args,
                                lower_case, tokenizer, port, timeout
            """
            r = pool.apply_async(self._build_topology_process,
                                 args=(data_items[start_index:end_index], self.topology_builder, self.graph_type,
                                       self.dynamic_graph_type, self.dynamic_init_topology_builder,
                                       self.merge_strategy, self.edge_strategy, self.dynamic_init_topology_aux_args,
                                       self.lower_case, self.tokenizer, self.port, self.timeout))
            res_l.append(r)
        pool.close()
        pool.join()

        data_items = []
        for i in range(thread_number):
            # start_index = total * i // thread_number
            # end_index = total * (i + 1) // thread_number

            res = res_l[i].get()
            # datas = data_items[start_index:end_index]
            for data in res:
                data.graph = data.graph.to(self.device)
                data_items.append(data)

        # if len([x for x in data_items if not hasattr(x, 'graph')])>0:
        #     a = 0
        return data_items

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

    def download(self):
        return

    def build_vocab(self):
        data_for_vocab = self.train
        if self.use_val_for_vocab:
            data_for_vocab = data_for_vocab + self.val

        vocab_model = VocabModel.build(saved_vocab_file=self.processed_file_paths['vocab'],
                                       data_set=data_for_vocab,
                                       tokenizer=self.tokenizer,
                                       lower_case=self.lower_case,
                                       max_word_vocab_size=None,
                                       min_word_vocab_freq=3,
                                       pretrained_word_emb_file=self.pretrained_word_emb_file,
                                       word_emb_size=self.word_emb_size,
                                       share_vocab=self.share_vocab)
        self.vocab_model = vocab_model
        return self.vocab_model

    def parse_file(self, file_path):
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
                input = ' '.join(' '.join(example_dict['article']).split()[:400]).lower()
                output = ' '.join(' '.join(['<t> ' + sent[0] + ' . </t>' for sent in example_dict['highlight']]).split()[:99]).lower()
                if input=='' or output=='':
                    continue
                data_item = Text2TextDataItem(input_text=input, output_text=output, tokenizer=self.tokenizer,
                                              share_vocab=self.share_vocab)
                data.append(data_item)
        return data
