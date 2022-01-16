import copy
import math
import warnings
import torch

from graph4nlp.pytorch.data.dataset import Text2TextDataItem, Text2TextDataset
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.utils.generic_utils import all_to_cuda

from .base import InferenceWrapperBase


class GeneratorInferenceWrapper(InferenceWrapperBase):
    def __init__(
        self,
        cfg,
        model,
        dataset=Text2TextDataset,
        data_item=Text2TextDataItem,
        topology_builder=None,
        dynamic_init_topology_builder=None,
        beam_size=3,
        topk=1,
        lower_case=True,
        tokenizer=None,
        share_vocab=True,
        **kwargs,
    ):
        """
            The inference wrapper for generation tasks.
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
            The initial graph topology builder. It is used to custermize your own graph\
                 construction method. We will set the default topology builder for you \
                 if it is ``None`` according to ``graph_name`` in ``cfg``.
        dynamic_init_topology_builder: GraphConstructionBase, default=None
            The dynamic initial graph topology builder. It is used to custermize your own \
                graph construction method. We will set the default topology builder for you\
                if it is ``None`` according to ``dynamic_init_graph_name`` in ``cfg``.
        lower_case: bool, default=True
        tokenizer: function, default=nltk.word_tokenize
        beam_size: int, default=3
        """
        super().__init__(
            cfg=cfg,
            model=model,
            topology_builder=topology_builder,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            lower_case=lower_case,
            tokenizer=tokenizer,
            dataset=dataset,
            data_item=data_item,
            beam_size=beam_size,
            topk=topk,
            share_vocab=share_vocab,
            **kwargs,
        )

        self.vocab_model = model.vocab_model
        self.use_copy = self.cfg["decoder_args"]["rnn_decoder_share"]["use_copy"]

    @torch.no_grad()
    def predict(self, raw_contents: list, batch_size=1):
        """
            Do the inference.
        Parameters
        ----------
        raw_contents: list
            The raw inputs. Example: ["sentence1", "sentence2"]
        batch_size: int, default=1
            The batch size of the inference.
        Returns
        -------
        Inference_results: object
            It will be the post-processed results. Examples: ["output1", "output2"]
        """
        # step 1: construct graph
        if len(raw_contents) == 0:
            warnings.warn("The input ``raw_contents`` is empty.")
            return []

        if batch_size <= 0:
            raise ValueError("``batch_size`` should be > 0")

        if len(raw_contents) < batch_size:
            batch_size = len(raw_contents)

        ret_collect = []

        for i in range(math.ceil(len(raw_contents) / batch_size)):
            data_collect = raw_contents[i * batch_size : (i + 1) * batch_size]

            data_items = []
            vocab_model = copy.deepcopy(self.vocab_model)
            device = next(self.parameters()).device
            data_items = self.preprocess(raw_contents=data_collect)

            collate_data = self.dataset.collate_fn(data_items)
            collate_data = all_to_cuda(collate_data, device)

            # forward
            if self.use_copy:
                oov_dict = prepare_ext_vocab(
                    batch_graph=collate_data["graph_data"], vocab=vocab_model, device=device
                )
                ref_dict = oov_dict
            else:
                oov_dict = None
                ref_dict = self.vocab_model.out_word_vocab
            ret = self.model.inference_forward(
                collate_data, self.beam_size, topk=self.topk, oov_dict=oov_dict
            )
            ret = self.model.post_process(decode_results=ret, vocab=ref_dict)
            ret_collect.extend(ret)

        return ret_collect
