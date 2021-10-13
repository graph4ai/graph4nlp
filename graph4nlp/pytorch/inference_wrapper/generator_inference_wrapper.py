import torch
import torch.nn as nn
import copy
from graph4nlp.pytorch.data.dataset import Text2TextDataItem, Text2TextDataset
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab
from graph4nlp.pytorch.modules.utils.generic_utils import wordid2str


class GeneratorInferenceWrapper(nn.Module):
    def __init__(self, cfg, model, beam_size=3, lower_case=True, tokenizer=None):
        super().__init__()
        # cfg is expected to be fixed
        # TODO: lower_case and tokenizer should be removed
        self.cfg = cfg
        self.model = model
        self.vocab_model = model.vocab_model
        self.graph_type = model.graph_type
        self.beam_size = beam_size
        self.lower_case = lower_case
        self.use_copy = self.cfg["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.port = self.cfg["graph_construction_args"]["graph_construction_share"]["port"]
        self.timeout = self.cfg["graph_construction_args"]["graph_construction_share"]["timeout"]
        self.tokenizer = tokenizer
        self.merge_strategy = self.cfg["graph_construction_args"]["graph_construction_private"]["merge_strategy"]
        self.edge_strategy = self.cfg["graph_construction_args"]["graph_construction_private"]["edge_strategy"]
        
    
    def predict(self, raw_contents=["sentence1", "sentence2"]):
        # step 1: construct graph
        data_items = []
        vocab_model = copy.deepcopy(self.vocab_model)
        device = next(self.parameters()).device
        for raw_sentence in raw_contents:
            data_item = Text2TextDataItem(input_text=raw_sentence, output_text=None, tokenizer=self.tokenizer)
            if self.graph_type in ["dependency", "constituency", "ie"]:
                graph_type = "static"

            from graph4nlp.pytorch.modules.graph_construction import DependencyBasedGraphConstruction
            data_item = Text2TextDataset._build_topology_process(data_items=[data_item], topology_builder=DependencyBasedGraphConstruction, # self.graph_topology, 
                graph_type=graph_type, dynamic_graph_type=None, dynamic_init_topology_builder=None, merge_strategy=self.merge_strategy, edge_strategy=self.edge_strategy,
                dynamic_init_topology_aux_args=None, lower_case=self.lower_case, port=self.port, timeout=self.timeout, tokenizer=self.tokenizer)  # only support static graph types
            data_item = Text2TextDataset._vectorize_one_dataitem(data_item[0], self.vocab_model, use_ie=False) # not support IE
            data_items.append(data_item)
        collate_data = Text2TextDataset.collate_fn(data_items)
        batch_graph = collate_data["graph_data"].to(device)

        # forward
        if self.use_copy:
            oov_dict = prepare_ext_vocab(
                batch_graph=batch_graph, vocab=vocab_model, device=device
            )
            ref_dict = oov_dict
        else:
            oov_dict = None
            ref_dict = self.vocab.out_word_vocab
                
        ret = self.model.decoding_forward(batch_graph=batch_graph, beam_size=self.beam_size, oov_dict=oov_dict)

        return self.model.post_process(decode_results=ret, vocab=ref_dict)
