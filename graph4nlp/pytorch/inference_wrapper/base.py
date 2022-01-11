import json
from typing import Union
import torch.nn as nn

from graph4nlp.pytorch.data.dataset import (
    DataItem,
    Dataset,
    DoubleText2TextDataItem,
    KGCompletionDataItem,
    Text2LabelDataItem,
    Text2TreeDataItem,
    word_tokenize,
)
from graph4nlp.pytorch.modules.graph_construction.base import (
    DynamicGraphConstructionBase,
    StaticGraphConstructionBase,
)


class InferenceWrapperBase(nn.Module):
    def __init__(
        self,
        cfg: dict,
        model: nn.Module,
        dataset: Dataset,
        data_item: DataItem,
        topology_builder: Union[StaticGraphConstructionBase, DynamicGraphConstructionBase] = None,
        dynamic_init_topology_builder: StaticGraphConstructionBase = None,
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
        topology_builder: Union[StaticGraphConstructionBase, DynamicGraphConstructionBase], \
                            default=None
            The initial graph topology builder. It is used to custermize your own graph\
                 construction method. We will set the default topology builder for you \
                 if it is ``None`` according to ``graph_name`` in ``cfg``.
        dynamic_init_topology_builder: StaticGraphConstructionBase, default=None
            The dynamic initial graph topology builder. It is used to custermize your own \
                graph construction method. We will set the default topology builder for you\
                if it is ``None`` according to ``dynamic_init_graph_name`` in ``cfg``.
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
        self.dynamic_init_graph_name = cfg["graph_construction_args"][
            "graph_construction_private"
        ].get("dynamic_init_graph_name", None)
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
            port=self.port,
            graph_name=self.graph_name,
            dynamic_init_graph_name=self.dynamic_init_graph_name,
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
            elif self.data_item_class == KGCompletionDataItem:
                assert isinstance(raw_sentence, tuple)
                line = json.loads(raw_sentence[0])
                e1, rel, e2, rel_eval, e2_multi1, e2_multi2 = (
                    line["e1"],
                    line["rel"],
                    line["e2"],
                    line["rel_eval"],
                    line["e2_multi1"],
                    line["e2_multi2"],
                )
                data_item = self.data_item_class(
                    e1, rel, e2, rel_eval, e2_multi1, e2_multi2, tokenizer=self.tokenizer
                )
            elif self.data_item_class == Text2TreeDataItem:
                data_item = self.data_item_class(
                    input_text=raw_sentence,
                    output_text=None,
                    output_tree=None,
                    tokenizer=self.tokenizer,
                )
            elif self.data_item_class == Text2LabelDataItem:
                data_item = self.data_item_class(input_text=raw_sentence, tokenizer=self.tokenizer)
            else:
                data_item = self.data_item_class(
                    input_text=raw_sentence, tokenizer=self.tokenizer, output_text=None
                )

            data_item = self.dataset.process_data_items(data_items=[data_item])
            data_item = self.dataset._vectorize_one_dataitem(
                data_item[0], self.vocab_model, use_ie=use_ie
            )
            processed_data_items.append(data_item)
        return processed_data_items
