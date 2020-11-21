import copy
import torch
import numpy as np

from .padding_utils import pad_2d_vals_no_size
from ...data.data import GraphData
from .vocab_utils import VocabModel, Vocab


def prepare_ext_vocab(graph_list, vocab, gt_str=None, device=None):
    """
        The function for copy mechanism.
        It will find the oov dict in ``gt_str`` and build ``oov_dict``.
        And the tgt_seq will be built by ``oov_dict``.
        Note that, the word_index in ``oov_dict`` is saved in ``token_id_oov`` of GraphData.
    Parameters
    ----------
    graph_list: list[GraphData]
        The graph list.
    vocab: VocabModel
        The VocabModel.
    gt_str: list[str], default=None
        The list contains the raw target (i.e. the ground-truth for most task) sequence.
        Note that the raw sequence should not contains <UNK>.
        It will be processed by ``oov_dict`` and generate the ``tgt_seq``.
    device: torch.device, default=None
        The device.
    Returns
    -------
    oov_dict: Vocab
        The vocab contains the out-of-vocabulary words in graph_list.
    tgt_seq: torch.Tensor
        If the ``gt_str`` is not ``None``, it will be calculated.
        We process the ``gt_str`` with ``oov_dict``.
    """
    oov_dict = copy.deepcopy(vocab.in_word_vocab)
    for g in graph_list:
        token_matrix = []
        for node_idx in range(g.get_node_num()):
            node_token = g.node_attributes[node_idx]['token']
            if oov_dict.getIndex(node_token) == oov_dict.UNK:
                oov_dict._add_words(node_token)
            token_matrix.append([oov_dict.getIndex(node_token)])
        token_matrix = torch.tensor(token_matrix, dtype=torch.long).to(device)
        g.node_features['token_id_oov'] = token_matrix

    if gt_str is not None:
        oov_tgt_collect = []
        for s in gt_str:
            oov_tgt = oov_dict.to_index_sequence(s)
            oov_tgt.append(oov_dict.EOS)
            oov_tgt = np.array(oov_tgt)
            oov_tgt_collect.append(oov_tgt)

        output_pad = pad_2d_vals_no_size(oov_tgt_collect)

        tgt_seq = torch.from_numpy(output_pad).long().to(device)
        return oov_dict, tgt_seq
    else:
        return oov_dict