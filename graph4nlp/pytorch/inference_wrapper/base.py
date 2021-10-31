import torch.nn as nn

from graph4nlp.pytorch.data.dataset import DataItem, Dataset, DoubleText2TextDataItem, word_tokenize
from graph4nlp.pytorch.modules.graph_construction.base import GraphConstructionBase


class InferenceWrapperBase(nn.Module):
    def __init__(
        self,
        cfg: dict,
        model: nn.Module,
        dataset: Dataset,
        data_item: DataItem,
        topology_builder: GraphConstructionBase = None,
        dynamic_init_topology_builder: GraphConstructionBase = None,
        lower_case: bool = True,
        tokenizer=word_tokenize,
        **kwargs
    ):
        """
            The base class for inference wrapper.
        Parameters
        ----------
        cfg: dict
            The configure dictionary.
        model: nn.Module
            The model checkpoint.
            The model must support the following attributes:
                model.graph_name: str,
                    The graph type, eg: "dependency".
            The model must support the following api:
                model.inference_forward(batch_graph, **kwargs)
                    It is the forward process during inference.
                model.post_process()
                    It is the post-process method.
            The inference wrapper will do the pipeline as follows:
                1. model.inference_forward()
                2. model.post_process()
        dataset: Dataset,
            The dataset class.
        data_item: DataItem,
            The data_item class.
        topology_builder: GraphConstructionBase, default=None
            The initial graph topology builder. We will set the default topology builder for you
                if it is ``None`` according to ``graph_name`` in ``cfg``.
        dynamic_init_topology_builder: GraphConstructionBase, default=None
            The dynamic initial graph topology builder. We will set the default topology builder
                for you if it is ``None`` according to ``dynamic_init_graph_type`` in ``cfg``.
        lower_case: bool, default=True
            TBD: move it to template
        tokenizer: function, default=nltk.word_tokenize
        """
        super().__init__()
        # cfg is expected to be fixed
        # TODO: lower_case and tokenizer should be removed
        self.cfg = cfg
        self.model = model
        self.graph_name = model.graph_name
        self.dynamic_init_graph_type = cfg["graph_construction_args"][
            "graph_construction_private"
        ].get("dynamic_init_graph_type", None)
        self.lower_case = lower_case
        self.topology_builder = topology_builder
        self.dynamic_init_topology_builder = dynamic_init_topology_builder
        self.tokenizer = tokenizer

        self.port = self.cfg["graph_construction_args"]["graph_construction_share"]["port"]
        self.timeout = self.cfg["graph_construction_args"]["graph_construction_share"]["timeout"]

        self.merge_strategy = self.cfg["graph_construction_args"]["graph_construction_private"][
            "merge_strategy"
        ]
        self.edge_strategy = self.cfg["graph_construction_args"]["graph_construction_private"][
            "edge_strategy"
        ]

        self.dataset = dataset(
            graph_name=self.graph_name,
            dynamic_init_graph_type=self.dynamic_init_graph_type,
            topology_builder=topology_builder,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            lower_case=lower_case,
            tokenizer=tokenizer,
            merge_strategy=self.merge_strategy,
            edge_strategy=self.edge_strategy,
        )
        self.data_item_class = data_item
        for k, v in kwargs.items():
            setattr(self, k, v)

    def preprocess(self, raw_contents: list):
        processed_data_items = []
        use_ie = self.graph_name == "ie"  # hard code
        for raw_sentence in raw_contents:
            if self.data_item_class == DoubleText2TextDataItem:
                assert isinstance(
                    raw_sentence, (tuple, list)
                ), "Expect two inputs for DoubleText2TextDataItem, only one found."
                data_item = self.data_item_class(
                    input_text=raw_sentence[0],
                    input_text2=raw_sentence[1],
                    output_text=None,
                    tokenizer=self.tokenizer,
                    share_vocab=self.share_vocab,
                )
            else:
                data_item = self.data_item_class(
                    input_text=raw_sentence, output_text=None, tokenizer=self.tokenizer
                )

            data_item = self.dataset.process_data_items(data_items=[data_item])
            data_item = self.dataset._vectorize_one_dataitem(
                data_item[0], self.vocab_model, use_ie=use_ie
            )
            processed_data_items.append(data_item)
        return processed_data_items
