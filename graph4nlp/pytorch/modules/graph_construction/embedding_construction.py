import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from ..utils.torch_utils import to_cuda


class EmbeddingConstructionBase(nn.Module):
    """
    Base class for (initial) graph embedding construction.

    ...

    Attributes
    ----------


    Methods
    -------
    forward(feat)
        Generate initial node and/or edge embeddings for the input graph.
    """

    def __init__(self):
        super(EmbeddingConstructionBase, self).__init__()

    def forward(self):
        raise NotImplementedError()


class EmbeddingConstruction(EmbeddingConstructionBase):
    """
    Graph embedding construction class.

    ...

    Attributes
    ----------
    word_vocab: Vocab class
        Word vocab instance.

    word_emb_type: str or list of str
        Specify pretrained word embedding types. ``w2v`` includes GloVe, Word2Vec and etc. ``bert`` is to be supported.

    node_edge_level_emb_type: str
        Specify node/edge level embedding initialization strategies (e.g., ``mean``, ``lstm`` and ``bilstm``).

    graph_level_emb_type: str
        Specify graph level embedding initialization strategies (e.g., ``identity``, ``lstm``  and ``bilstm``).

    hidden_size: int
        Hidden size.

    fix_word_emb: boolean, default: ``True``
        Specify whether to fix pretrained word embeddings.

    dropout: float
        Dropout ratio


    Methods
    -------
    forward(feat)
        Generate initial node and/or edge embeddings for the input graph.
    """

    def __init__(self, word_vocab, word_emb_type,
                        node_edge_level_emb_type,
                        graph_level_emb_type,
                        hidden_size,
                        fix_word_emb=True,
                        dropout=None,
                        use_cuda=True):
        super(EmbeddingConstruction, self).__init__()
        self.node_edge_level_emb_type = node_edge_level_emb_type
        self.graph_level_emb_type = graph_level_emb_type

        if isinstance(word_emb_type, str):
            word_emb_type = [word_emb_type]

        self.word_embs = nn.ModuleList()
        if 'w2v' in word_emb_type:
            self.word_embs.append(WordEmbedding(
                            word_vocab.embeddings.shape[0],
                            word_vocab.embeddings.shape[1],
                            pretrained_word_emb=word_vocab.embeddings,
                            fix_word_emb=fix_word_emb))

        if 'bert' in word_emb_type:
            self.word_embs.append(BertEmbedding(fix_word_emb))

        if node_edge_level_emb_type == 'mean':
            self.node_level_emb = MeanEmbedding()

        elif node_edge_level_emb_type == 'lstm':
            self.node_level_emb = LSTMEmbedding(
                                    word_vocab.embeddings.shape[1],
                                    hidden_size, dropout=dropout,
                                    bidirectional=False,
                                    rnn_type='lstm', use_cuda=use_cuda)

        elif node_edge_level_emb_type == 'bilstm':
            self.node_level_emb = LSTMEmbedding(
                                    word_vocab.embeddings.shape[1],
                                    hidden_size, dropout=dropout,
                                    bidirectional=True,
                                    rnn_type='lstm', use_cuda=use_cuda)
        else:
            raise RuntimeError('Unknown node_edge_level_emb_type: {}'.format(node_edge_level_emb_type))

        if graph_level_emb_type == 'identity':
            self.graph_level_emb = None

        elif graph_level_emb_type == 'lstm':
            self.graph_level_emb = LSTMEmbedding(
                                    word_vocab.embeddings.shape[1] \
                                    if node_edge_level_emb_type == 'mean' else hidden_size,
                                    hidden_size, dropout=dropout,
                                    bidirectional=False,
                                    rnn_type='lstm', use_cuda=use_cuda)

        elif graph_level_emb_type == 'bilstm':
            self.graph_level_emb = LSTMEmbedding(
                                    word_vocab.embeddings.shape[1] \
                                    if node_edge_level_emb_type == 'mean' else hidden_size,
                                    hidden_size, dropout=dropout,
                                    bidirectional=True,
                                    rnn_type='lstm', use_cuda=use_cuda)
        else:
            raise RuntimeError('Unknown graph_level_emb_type: {}'.format(graph_level_emb_type))

    def forward(self, input_tensor, node_size, graph_size):
        feat = []
        for word_emb in self.word_embs:
            feat.append(word_emb(input_tensor))

        feat = torch.cat(feat, dim=-1)

        feat = self.node_level_emb(feat, node_size)
        if self.node_edge_level_emb_type in ('lstm', 'bilstm'):
            feat = feat[-1]

        if self.graph_level_emb is not None:
            feat = self.graph_level_emb(torch.unsqueeze(feat, 0), graph_size)
            if self.graph_level_emb_type in ('lstm', 'bilstm'):
                feat = feat[0]

            feat = torch.squeeze(feat, 0)

        return feat

class WordEmbedding(nn.Module):
    """
    Word embedding layer.

    ...

    Attributes
    ----------


    Methods
    -------
    forward(input_tensor)
        Return word embeddings.


    Examples
    ------
    word_emb = WordEmbedding(vocab_size, emb_size, padding_idx=0, pretrained_word_emb=None, fix_word_emb=False)
    """

    def __init__(self, vocab_size, emb_size, padding_idx=0, pretrained_word_emb=None, fix_word_emb=False):
        super(WordEmbedding, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx,
                            _weight=torch.from_numpy(pretrained_word_emb).float()
                            if pretrained_word_emb is not None else None)

        if fix_word_emb:
            print('[ Fix word embeddings ]')
            for param in self.word_emb.parameters():
                param.requires_grad = False

    def forward(self, input_tensor):
        emb = self.word_emb(input_tensor)

        return emb


class BertEmbedding(nn.Module):
    """
    BERT embedding layer.

    ...

    Attributes
    ----------


    Methods
    -------
    forward(feat)
        Generate BERT embeddings.
    """
    def __init__(self):
        super(BertEmbedding, self).__init__()

    def forward(self, feat):
        raise NotImplementedError()

class MeanEmbedding(nn.Module):
    """
    Mean embedding layer.

    ...

    Attributes
    ----------


    Methods
    -------
    forward(emb)
        Return the input embeddings.
    """
    def __init__(self):
        super(MeanEmbedding, self).__init__()

    def forward(self, emb, len_):
        return torch.sum(emb, dim=-2) / len_


class LSTMEmbedding(nn.Module):
    """
    LSTM embedding layer.

    ...

    Attributes
    ----------


    Methods
    -------
    forward(feat)
        Apply the LSTM network to the input sequence of embeddings.
    """
    def __init__(self, input_size, hidden_size,
                        dropout=None, bidirectional=False,
                        rnn_type='lstm', use_cuda=True):
        super(LSTMEmbedding, self).__init__()
        if not rnn_type in ('lstm', 'gru'):
            raise RuntimeError('rnn_type is expected to be lstm or gru, got {}'.format(rnn_type))

        # if bidirectional:
        #     print('[ Using bidirectional {} encoder ]'.format(rnn_type))

        # else:
        #     print('[ Using {} encoder ]'.format(rnn_type))

        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')

        self.dropout = dropout
        self.rnn_type = rnn_type
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(input_size, self.hidden_size, 1, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, x_len):
        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)

        h0 = to_cuda(torch.zeros(self.num_directions, x_len.size(0), self.hidden_size), self.use_cuda)
        if self.rnn_type == 'lstm':
            c0 = to_cuda(torch.zeros(self.num_directions, x_len.size(0), self.hidden_size), self.use_cuda)
            packed_h, (packed_h_t, _) = self.model(x, (h0, c0))
            if self.num_directions == 2:
                packed_h_t = torch.cat([packed_h_t[i] for i in range(packed_h_t.size(0))], -1)
            else:
                packed_h_t = packed_h_t.squeeze(0)

        else:
            packed_h, packed_h_t = self.model(x, h0)
            if self.num_directions == 2:
                packed_h_t = packed_h_t.transpose(0, 1).contiguous().view(query_lengths.size(0), -1)
            else:
                packed_h_t = packed_h_t.squeeze(0)

        hh, _ = pad_packed_sequence(packed_h, batch_first=True)

        # restore the sorting
        _, inverse_indx = torch.sort(indx, 0)
        restore_hh = hh[inverse_indx]
        restore_packed_h_t = packed_h_t[inverse_indx]

        return restore_hh, restore_packed_h_t


if __name__ == '__main__':
    # For test purpose
    from ..utils.vocab_utils import VocabModel
    from ..utils.padding_utils import pad_2d_vals_no_size


    raw_text_data = [['I like nlp.', 'Same here!'], ['I like graph.', 'Same here!']]

    vocab_model = VocabModel(raw_text_data, max_word_vocab_size=None,
                                min_word_vocab_freq=1,
                                word_emb_size=300)

    src_text_seq = list(zip(*raw_text_data))[0]
    src_idx_seq = [vocab_model.word_vocab.to_index_sequence(each) for each in src_text_seq]
    src_len = torch.LongTensor([len(each) for each in src_idx_seq])
    num_seq = torch.LongTensor([len(src_len)])
    input_tensor = torch.LongTensor(pad_2d_vals_no_size(src_idx_seq))
    print('input_tensor: {}'.format(input_tensor.shape))

    emb_constructor = EmbeddingConstruction(vocab_model.word_vocab, 'w2v', 'bilstm', 'bilstm', 128)
    emb = emb_constructor(input_tensor, src_len, num_seq)
    print('emb: {}'.format(emb.shape))
