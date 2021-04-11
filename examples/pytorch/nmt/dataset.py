from graph4nlp.pytorch.data.dataset import Text2TextDataItem, Text2TextDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
import torch
import os
import json
from stanfordcorenlp import StanfordCoreNLP
import pickle
import re
import nltk
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size



class IWSLT14Dataset(Text2TextDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'train.pkl', 'val': "val.pkl", 'test': 'test.pkl'}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {'vocab': 'vocab.pt', 'data': 'data.pt'}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded. Please prepare the raw data yourself.')
        return
    
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
        with open(file_path, 'rb') as f:
            lines = pickle.load(f)
        for line in lines:
            input, output = line
            if input.strip() == "" or output.strip() == "":
                continue
            input_len = len(input.split())
            output_len = len(output.split())
            if input_len > 50 or output_len > 50:
                continue
            data_item = Text2TextDataItem(input_text=input, output_text=output, tokenizer=self.tokenizer,
                                            share_vocab=self.share_vocab)
            data.append(data_item)
        return data

    

    def __init__(self, root_dir,
                 topology_builder, topology_subdir,
                 tokenizer=nltk.RegexpTokenizer(" ", gaps=True).tokenize,
                 pretrained_word_emb_file=None,
                 val_split_ratio=None,
                 use_val_for_vocab=False,
                 graph_type='static',
                 merge_strategy="tailhead", edge_strategy=None,
                 seed=None,
                 word_emb_size=300, share_vocab=False,
                 dynamic_graph_type=None,
                 dynamic_init_topology_builder=None,
                 dynamic_init_topology_aux_args=None):
        """

        Parameters
        ----------
        root_dir: str
            The path of dataset.
        topology_builder: GraphConstructionBase
            The graph construction class.
        topology_subdir: str
            The directory name of processed path.
        graph_type: str, default='static'
            The graph type. Expected in ('static', 'dynamic')
        edge_strategy: str, default=None
            The edge strategy. Expected in (None, 'homogeneous', 'as_node'). If set `None`, it will be 'homogeneous'.
        merge_strategy: str, default=None
            The strategy to merge sub-graphs. Expected in (None, 'tailhead', 'user_define').
            If set `None`, it will be 'tailhead'.
        share_vocab: bool, default=False
            Whether to share the input vocabulary with the output vocabulary.
        dynamic_graph_type: str, default=None
            The dynamic graph type. It is only available when `graph_type` is set 'dynamic'.
            Expected in (None, 'node_emb', 'node_emb_refined').
        init_graph_type: str, default=None
            The initial graph topology. It is only available when `graph_type` is set 'dynamic'.
            Expected in (None, 'dependency', 'constituency')
        """
        # Initialize the dataset. If the preprocessed files are not found, then do the preprocessing and save them.
        super(IWSLT14Dataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                            topology_subdir=topology_subdir, graph_type=graph_type,
                                            edge_strategy=edge_strategy, merge_strategy=merge_strategy,
                                            share_vocab=share_vocab, pretrained_word_emb_file=pretrained_word_emb_file,
                                            val_split_ratio=val_split_ratio, seed=seed, word_emb_size=word_emb_size,
                                            tokenizer=tokenizer,
                                            use_val_for_vocab=use_val_for_vocab,
                                            dynamic_graph_type=dynamic_graph_type,
                                            dynamic_init_topology_builder=dynamic_init_topology_builder,
                                            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args)

    @staticmethod
    def collate_fn(data_list):
        graph_data = [item.graph for item in data_list]
        from graph4nlp.pytorch.data.data import to_batch
        big_graph = to_batch(graph_data)

        output_numpy = [item.output_np for item in data_list]
        output_str = [item.output_text.lower().strip() for item in data_list]
        output_pad = pad_2d_vals_no_size(output_numpy)

        tgt_seq = torch.from_numpy(output_pad).long()
        # return [graph_data, tgt_seq, output_str]
        return {
            "graph_data": big_graph,
            "tgt_seq": tgt_seq,
            "output_str": output_str
        }



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


