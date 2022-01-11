import math
import warnings

from graph4nlp.pytorch.data.dataset import Text2LabelDataItem, Text2LabelDataset
from graph4nlp.pytorch.modules.utils.generic_utils import all_to_cuda

from .base import InferenceWrapperBase


class ClassifierInferenceWrapper(InferenceWrapperBase):
    def __init__(
        self,
        cfg,
        model,
        dataset=Text2LabelDataset,
        data_item=Text2LabelDataItem,
        topology_builder=None,
        dynamic_init_topology_builder=None,
        lower_case=True,
        tokenizer=None,
        label_names=None,
    ):
        """
            The inference wrapper for classification tasks.
        Parameters
        ----------
        cfg: dict
            The configure dictionary.
        model: nn.Module
            The model checkpoint.
            The model must support the following attributes:
                model.graph_name: str,
                    The graph name, eg: "dependency".
            The model must support the following api:
                model.inference_forward(batch_graph, **kwargs)
                    It is the forward process during inference.
                model.post_process()
                    It is the post-process method.
            The inference wrapper will do the pipeline as follows:
                1. model.inference_forward()
                2. model.post_process()
            The output of the model.post_process should be the vector or matrix \
                to contain the index of the class
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
        classification_label: the label tags which maps the classifier index
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
        )

        self.label_names = label_names
        self.vocab_model = model.vocab_model
        self.data_item = data_item

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
            device = next(self.parameters()).device
            data_items = self.preprocess(raw_contents=data_collect)

            collate_data = self.dataset.collate_fn(data_items)
            collate_data = all_to_cuda(collate_data, device)

            # forward
            if self.data_item == "KGCompletionDataItem":
                KG_graph = data_collect[i][1]
                ret = self.model.inference_forward(collate_data, KG_graph)
                ret = self.model.post_process(logits=ret, e2=collate_data["e2_tensor"])
                print(
                    "e1 = {}, rel = {}, pred e2 = {}".format(
                        data_items[0].e1,
                        data_items[0].rel,
                        self.vocab_model.in_word_vocab.getWord(ret),
                    )
                )
                ret = [ret]
            else:
                ret = self.model.inference_forward(collate_data)
                ret = self.model.post_process(logits=ret, label_names=self.label_names)

            ret_collect.extend(ret)

        return ret_collect
