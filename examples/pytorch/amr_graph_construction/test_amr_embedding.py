from graph4nlp.pytorch.datasets.jobs import JobsDataset
from graph4nlp.pytorch.data.dataset import Text2TextDataset
from copy import deepcopy
import numpy as np
import torch.utils.data

from ..data.data import GraphData

from ..modules.utils.padding_utils import pad_2d_vals_no_size
from ..modules.utils.tree_utils import Vocab as VocabForTree

class AMRDataset(Text2TextDataset):
    """

        Parameters
        ----------
        root_dir: str
            The path of dataset.
        graph_construction_name: str
            The name of graph construction method. E.g., "dependency".
            Note that if it is in the provided graph names (i.e., "dependency", \
                "constituency", "ie", "node_emb", "node_emb_refine"), the following \
                parameters are set by default and users can't modify them:
                1. ``topology_builder``
                2. ``static_or_dynamic``
            If you need to customize your graph construction method, you should rename the \
                ``graph_construction_name`` and set the parameters above.
        topology_builder: GraphConstructionBase, default=None
            The graph construction class.
        topology_subdir: str
            The directory name of processed path.
        static_or_dynamic: str, default='static'
            The graph type. Expected in ('static', 'dynamic')
        edge_strategy: str, default=None
            The edge strategy. Expected in (None, 'homogeneous', 'as_node').
            If set `None`, it will be 'homogeneous'.
        merge_strategy: str, default=None
            The strategy to merge sub-graphs. Expected in (None, 'tailhead', 'user_define').
            If set `None`, it will be 'tailhead'.
        share_vocab: bool, default=False
            Whether to share the input vocabulary with the output vocabulary.
        dynamic_init_graph_name: str, default=None
            The graph name of the initial graph. Expected in (None, "line", "dependency", \
                "constituency").
            Note that if it is in the provided graph names (i.e., "line", "dependency", \
                "constituency"), the following parameters are set by default and users \
                can't modify them:
                1. ``dynamic_init_topology_builder``
            If you need to customize your graph construction method, you should rename the \
                ``graph_name`` and set the parameters above.
        dynamic_init_topology_builder: GraphConstructionBase
            The graph construction class.
        dynamic_init_topology_aux_args: None,
            TBD.
        """

    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {"train": "train.txt", "test": "test.txt"}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'split_ids'."""
        return {"vocab": "vocab.pt", "data": "data.pt"}

    def download(self):
        # raise NotImplementedError(
        #     'This dataset is now under test and cannot be downloaded.
        # Please prepare the raw data yourself.')
        return

    def __init__(
        self,
        root_dir,
        topology_subdir,
        graph_construction_name,
        static_or_dynamic="static",
        topology_builder=None,
        merge_strategy="tailhead",
        edge_strategy=None,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
        pretrained_word_emb_name="6B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=None,
        seed=None,
        word_emb_size=300,
        share_vocab=True,
        lower_case=True,
        thread_number=1,
        port=9000,
        for_inference=None,
        reused_vocab_model=None,
    ):
        # Initialize the dataset. If the preprocessed files are not found,
        # then do the preprocessing and save them.
        super(JobsDataset, self).__init__(
            root_dir=root_dir,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            graph_construction_name=graph_construction_name,
            static_or_dynamic=static_or_dynamic,
            edge_strategy=edge_strategy,
            merge_strategy=merge_strategy,
            share_vocab=share_vocab,
            lower_case=lower_case,
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            seed=seed,
            word_emb_size=word_emb_size,
            thread_number=thread_number,
            port=port,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
        )
    @classmethod
    def _vectorize_one_dataitem(cls, data_item, vocab_model, use_ie=False):

        item = deepcopy(data_item)
        graph: GraphData = item.graph
        token_matrix = []
        for node_idx in range(graph.get_node_num()):
            node_token = graph.node_attributes[node_idx]["token"]
            node_token_id = vocab_model.in_word_vocab.getIndex(node_token, use_ie)
            graph.node_attributes[node_idx]["token_id"] = node_token_id

            token_matrix.append([node_token_id])
        if use_ie:
            for i in range(len(token_matrix)):
                token_matrix[i] = np.array(token_matrix[i][0])
            token_matrix = pad_2d_vals_no_size(token_matrix)
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features["token_id"] = token_matrix
        else:
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features["token_id"] = token_matrix

        if use_ie and "token" in graph.edge_attributes[0].keys():
            edge_token_matrix = []
            for edge_idx in range(graph.get_edge_num()):
                edge_token = graph.edge_attributes[edge_idx]["token"]
                edge_token_id = vocab_model.in_word_vocab.getIndex(edge_token, use_ie)
                graph.edge_attributes[edge_idx]["token_id"] = edge_token_id
                edge_token_matrix.append([edge_token_id])
            if use_ie:
                for i in range(len(edge_token_matrix)):
                    edge_token_matrix[i] = np.array(edge_token_matrix[i][0])
                edge_token_matrix = pad_2d_vals_no_size(edge_token_matrix)
                edge_token_matrix = torch.tensor(edge_token_matrix, dtype=torch.long)
                graph.edge_features["token_id"] = edge_token_matrix

        tgt = item.output_text
        if isinstance(tgt, str):
            tgt_token_id = vocab_model.out_word_vocab.to_index_sequence(tgt)
            tgt_token_id.append(vocab_model.out_word_vocab.EOS)
            tgt_token_id = np.array(tgt_token_id)
            item.output_np = tgt_token_id
        return item

    def vectorization(self, data_items):
        for idx in range(len(data_items)):
            data_items[idx] = self._vectorize_one_dataitem(
                data_items[idx], self.vocab_model
            )
