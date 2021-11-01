import pickle
import nltk
import torch

from graph4nlp.pytorch.data.dataset import Text2TextDataItem, Text2TextDataset
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals_no_size


class IWSLT14Dataset(Text2TextDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        if self.for_inference:
            return {"test": "test.pkl"}
        else:
            return {"train": "train.pkl", "val": "val.pkl", "test": "test.pkl"}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {"vocab": "vocab.pt", "data": "data.pt"}

    # def download(self):
    #     raise NotImplementedError(
    #         "This dataset is now under test and cannot be downloaded. "
    #         "Please prepare the raw data yourself."
    #     )

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`.
        The file format is specified by each individual task-specific
        base class. Returns all the indices of data items in this file w.r.t. the whole dataset.

        For Text2TextDataset, the format of the input file should contain lines of input,
        each line representing one record of data. The input and output is separated by a tab(\t).

        Examples
        --------
        input: list job use languageid0 job ( ANS ) , language ( ANS , languageid0 )

        DataItem: input_text="list job use languageid0", output_text="job ( ANS ) ,
        language ( ANS , languageid0 )"

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
        with open(file_path, "rb") as f:
            lines = pickle.load(f)
        for line in lines:
            input, output = line
            if input.strip() == "" or output.strip() == "":
                continue
            input_len = len(input.split())
            output_len = len(output.split())
            if input_len > 50 or output_len > 50:
                continue
            data_item = Text2TextDataItem(
                input_text=input,
                output_text=output,
                tokenizer=self.tokenizer,
                share_vocab=self.share_vocab,
            )
            data.append(data_item)
        return data

    def __init__(
        self,
        graph_name,
        root_dir=None,
        topology_subdir=None,
        topology_builder=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        tokenizer=nltk.RegexpTokenizer(" ", gaps=True).tokenize,
        pretrained_word_emb_name=None,
        pretrained_word_emb_url=None,
        target_pretrained_word_emb_name=None,
        target_pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=".vector_cache/",
        val_split_ratio=None,
        use_val_for_vocab=False,
        merge_strategy="tailhead",
        edge_strategy=None,
        seed=None,
        word_emb_size=300,
        share_vocab=False,
        for_inference=False,
        reused_vocab_model=None,
        lower_case=True,
    ):
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
            The edge strategy. Expected in (None, 'homogeneous', 'as_node').
            If set `None`, it will be 'homogeneous'.
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
        # Initialize the dataset.
        # If the preprocessed files are not found, then do the preprocessing and save them.
        super(IWSLT14Dataset, self).__init__(
            root_dir=root_dir,
            topology_subdir=topology_subdir,
            graph_name=graph_name,
            topology_builder=topology_builder,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            share_vocab=share_vocab,
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            target_pretrained_word_emb_name=target_pretrained_word_emb_name,
            target_pretrained_word_emb_url=target_pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            val_split_ratio=val_split_ratio,
            seed=seed,
            word_emb_size=word_emb_size,
            tokenizer=tokenizer,
            use_val_for_vocab=use_val_for_vocab,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            lower_case=lower_case,
        )

    @staticmethod
    def collate_fn(data_list):
        graph_data = [item.graph for item in data_list]
        from graph4nlp.pytorch.data.data import to_batch

        big_graph = to_batch(graph_data)
        if isinstance(data_list[0].output_text, str):  # has ground truth
            output_numpy = [item.output_np for item in data_list]
            output_str = [item.output_text.lower().strip() for item in data_list]
            output_pad = pad_2d_vals_no_size(output_numpy)

            tgt_seq = torch.from_numpy(output_pad).long()
        else:
            output_str = []
            tgt_seq = None
        return {"graph_data": big_graph, "tgt_seq": tgt_seq, "output_str": output_str}
