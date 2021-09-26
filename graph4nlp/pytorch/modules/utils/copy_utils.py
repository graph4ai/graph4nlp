import copy
import numpy as np
import torch

from .padding_utils import pad_2d_vals_no_size


def prepare_ext_vocab(batch_graph, vocab, gt_str=None, device=None):
    """
        The function for copy mechanism.
        It will find the oov dict in ``gt_str`` and build ``oov_dict``.
        And the tgt_seq will be built by ``oov_dict``.
        Note that, the word_index in ``oov_dict`` is saved in ``token_id_oov`` of GraphData.
    Parameters
    ----------
    batch_graph: GraphData
        The graph.
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
        The vocab contains the out-of-vocabulary words in batch_graph.
    tgt_seq: torch.Tensor
        If the ``gt_str`` is not ``None``, it will be calculated.
        We process the ``gt_str`` with ``oov_dict``.
    """
    oov_dict = copy.deepcopy(vocab.in_word_vocab)
    token_matrix = batch_graph.node_features["token_id"].squeeze(1)
    unk_index = (
        (token_matrix == oov_dict.UNK).nonzero(as_tuple=False).squeeze(1).detach().cpu().numpy()
    )
    unk_token = [batch_graph.node_attributes[index]["token"] for index in unk_index]
    oov_dict._add_words(unk_token)
    token_matrix_oov = token_matrix.clone()
    for idx in unk_index:
        unk_token = batch_graph.node_attributes[idx]["token"]
        oov_dict._add_words(unk_token)
        token_matrix_oov[idx] = oov_dict.getIndex(unk_token)
    batch_graph.node_features["token_id_oov"] = token_matrix_oov

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
