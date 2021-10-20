import copy

from graph4nlp.pytorch.data.dataset import Text2TextDataItem, Text2TextDataset
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab

from .base import InferenceWrapperBase


class GeneratorInferenceWrapper(InferenceWrapperBase):
    def __init__(
        self,
        cfg,
        model,
        dataset=Text2TextDataset,
        data_item=Text2TextDataItem,
        topology_builder=None,
        dynamic_topology_builder=None,
        beam_size=3,
        lower_case=True,
        tokenizer=None,
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
                model.graph_type: str,
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
            The initial graph topology builder. We will set the default topology builder \
                 for you if it is ``None`` according to ``graph_type`` in ``cfg``.
        dynamic_init_topology_builder: GraphConstructionBase, default=None
            The dynamic initial graph topology builder. We will set the default topology \
                 builder for you if it is ``None`` according to \
                 ``dynamic_init_graph_type`` in ``cfg``.
        lower_case: bool, default=True
        tokenizer: function, default=nltk.word_tokenize
        beam_size: int, default=3
        """
        super().__init__(
            cfg=cfg,
            model=model,
            topology_builder=topology_builder,
            dynamic_init_topology_builder=dynamic_topology_builder,
            lower_case=lower_case,
            tokenizer=tokenizer,
            dataset=dataset,
            data_item=data_item,
        )

        self.beam_size = beam_size
        self.vocab_model = model.vocab_model
        self.use_copy = self.cfg["decoder_args"]["rnn_decoder_share"]["use_copy"]

    def predict(self, raw_contents: list):
        """
            Do the inference.
        Parameters
        ----------
        raw_contents: list
            The raw inputs. Example: ["sentence1", "sentence2"]

        Returns
        -------
        Inference_results: object
            It will be the post-processed results. Examples: ["output1", "output2"]
        """
        # step 1: construct graph
        data_items = []
        vocab_model = copy.deepcopy(self.vocab_model)
        device = next(self.parameters()).device
        data_items = self.preprocess(raw_contents=raw_contents)

        collate_data = self.dataset.collate_fn(data_items)
        batch_graph = collate_data["graph_data"].to(device)

        # forward
        if self.use_copy:
            oov_dict = prepare_ext_vocab(batch_graph=batch_graph, vocab=vocab_model, device=device)
            ref_dict = oov_dict
        else:
            oov_dict = None
            ref_dict = self.vocab.out_word_vocab

        ret = self.model.inference_forward(
            batch_graph=batch_graph, beam_size=self.beam_size, oov_dict=oov_dict
        )

        return self.model.post_process(decode_results=ret, vocab=ref_dict)
