import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from ..utils.generic_utils import to_cuda

class EmbeddingConstructionBase(nn.Module):
    """
    Base class for (initial) graph embedding construction.

    ...

    Attributes
    ----------
    feat : dict
        Raw features of graph nodes and/or edges.

    Methods
    -------
    forward(raw_text_data)
        Generate dynamic graph topology and embeddings.
    """

    def __init__(self):
        super(EmbeddingConstructionBase, self).__init__()

    def forward(self):
        raise NotImplementedError()

class EmbeddingConstructionBase(nn.Module):
    """
    Base class for (initial) graph embedding construction.

    ...

    Attributes
    ----------
    feat : dict
        Raw features of graph nodes and/or edges.

    Methods
    -------
    forward(raw_text_data)
        Generate dynamic graph topology and embeddings.
    """

    def __init__(self):
        super(EmbeddingConstructionBase, self).__init__()

    def forward(self):
        raise NotImplementedError()

class EmbeddingConstructionBase(nn.Module):
    """Basic class for embedding construction.
    """
    def __init__(self):
        super(EmbeddingConstructionBase, self).__init__()

    def forward(self):
        """Compute initial node/edge embeddings.

        Raises
        ------
        NotImplementedError
            NotImplementedError.
        """
        raise NotImplementedError()

class EmbeddingConstruction(EmbeddingConstructionBase):
    """Initial graph embedding construction class.

    Parameters
    ----------
    word_vocab : Vocab
        The word vocabulary.
    word_emb_type : str or list of str
        Specify pretrained word embedding types including "w2v" and/or "bert".
    node_edge_emb_strategy : str
        Specify node/edge embedding strategies including "mean", "lstm",
        "gru", "bilstm" and "bigru".
    seq_info_encode_strategy : str
        Specify strategies of encoding sequential information in raw text
        data including "none", "lstm", "gru", "bilstm" and "bigru". You might
        want to do this in some situations, e.g., when all the nodes are single
        tokens extracted from the raw text.
    hidden_size : int, optional
        The hidden size of RNN layer, default: ``None``.
    fix_word_emb : boolean, optional
        Specify whether to fix pretrained word embeddings, default: ``True``.
    dropout : float, optional
        Dropout ratio, default: ``None``.
    device : torch.device, optional
        Specify computation device (e.g., CPU), default: ``None`` for using CPU.
    """
    def __init__(self, word_vocab, word_emb_type,
                        node_edge_emb_strategy,
                        seq_info_encode_strategy,
                        hidden_size=None,
                        fix_word_emb=True,
                        dropout=None,
                        device=None):
        super(EmbeddingConstruction, self).__init__()
        self.node_edge_emb_strategy = node_edge_emb_strategy
        self.seq_info_encode_strategy = seq_info_encode_strategy

        if isinstance(word_emb_type, str):
            word_emb_type = [word_emb_type]

        self.word_emb_layers = nn.ModuleList()
        if 'w2v' in word_emb_type:
            self.word_emb_layers.append(WordEmbedding(
                            word_vocab.embeddings.shape[0],
                            word_vocab.embeddings.shape[1],
                            pretrained_word_emb=word_vocab.embeddings,
                            fix_word_emb=fix_word_emb))

        if 'bert' in word_emb_type:
            self.word_emb_layers.append(BertEmbedding(fix_word_emb))

        if node_edge_emb_strategy == 'mean':
            self.node_edge_emb_layer = MeanEmbedding()
        elif node_edge_emb_strategy == 'lstm':
            self.node_edge_emb_layer = RNNEmbedding(
                                    word_vocab.embeddings.shape[1],
                                    hidden_size, dropout=dropout,
                                    bidirectional=False,
                                    rnn_type='lstm', device=device)
        elif node_edge_emb_strategy == 'bilstm':
            self.node_edge_emb_layer = RNNEmbedding(
                                    word_vocab.embeddings.shape[1],
                                    hidden_size, dropout=dropout,
                                    bidirectional=True,
                                    rnn_type='lstm', device=device)
        elif node_edge_emb_strategy == 'gru':
            self.node_edge_emb_layer = RNNEmbedding(
                                    word_vocab.embeddings.shape[1],
                                    hidden_size, dropout=dropout,
                                    bidirectional=False,
                                    rnn_type='gru', device=device)
        elif node_edge_emb_strategy == 'bigru':
            self.node_edge_emb_layer = RNNEmbedding(
                                    word_vocab.embeddings.shape[1],
                                    hidden_size, dropout=dropout,
                                    bidirectional=True,
                                    rnn_type='gru', device=device)
        else:
            raise RuntimeError('Unknown node_edge_emb_strategy: {}'.format(node_edge_emb_strategy))

        if seq_info_encode_strategy == 'none':
            self.seq_info_encode_layer = None
        elif seq_info_encode_strategy == 'lstm':
            self.seq_info_encode_layer = RNNEmbedding(
                                    word_vocab.embeddings.shape[1] \
                                    if node_edge_emb_strategy == 'mean' else hidden_size,
                                    hidden_size, dropout=dropout,
                                    bidirectional=False,
                                    rnn_type='lstm', device=device)
        elif seq_info_encode_strategy == 'bilstm':
            self.seq_info_encode_layer = RNNEmbedding(
                                    word_vocab.embeddings.shape[1] \
                                    if node_edge_emb_strategy == 'mean' else hidden_size,
                                    hidden_size, dropout=dropout,
                                    bidirectional=True,
                                    rnn_type='lstm', device=device)
        elif seq_info_encode_strategy == 'gru':
            self.seq_info_encode_layer = RNNEmbedding(
                                    word_vocab.embeddings.shape[1] \
                                    if node_edge_emb_strategy == 'mean' else hidden_size,
                                    hidden_size, dropout=dropout,
                                    bidirectional=False,
                                    rnn_type='gru', device=device)
        elif seq_info_encode_strategy == 'bigru':
            self.seq_info_encode_layer = RNNEmbedding(
                                    word_vocab.embeddings.shape[1] \
                                    if node_edge_emb_strategy == 'mean' else hidden_size,
                                    hidden_size, dropout=dropout,
                                    bidirectional=True,
                                    rnn_type='gru', device=device)
        else:
            raise RuntimeError('Unknown seq_info_encode_strategy: {}'.format(seq_info_encode_strategy))

    def forward(self, input_tensor, item_size, num_items):
        """Compute initial node/edge embeddings.

        Parameters
        ----------
        input_tensor : torch.LongTensor
            The input word index sequence, shape: [num_items, max_size].
        item_size : torch.LongTensor
            The length of word sequence per node/edge, shape: [num_items].
        num_items : torch.LongTensor
            The number of nodes/edges, shape: [1].

        Returns
        -------
        torch.Tensor
            The initial node/edge embeddings.
        """
        feat = []
        for word_emb_layer in self.word_emb_layers:
            feat.append(word_emb_layer(input_tensor))

        feat = torch.cat(feat, dim=-1)

        feat = self.node_edge_emb_layer(feat, item_size)
        if self.node_edge_emb_strategy in ('lstm', 'bilstm', 'gru', 'bigru'):
            feat = feat[-1]

        if self.seq_info_encode_layer is not None:
            feat = self.seq_info_encode_layer(torch.unsqueeze(feat, 0), num_items)
            if self.seq_info_encode_strategy in ('lstm', 'bilstm', 'gru', 'bigru'):
                feat = feat[0]

            feat = torch.squeeze(feat, 0)

        return feat

class WordEmbedding(nn.Module):
    """Word embedding class.

    Parameters
    ----------
    vocab_size : int
        The word vocabulary size.
    emb_size : int
        The word embedding size.
    padding_idx : int, optional
        The padding index, default: ``0``.
    pretrained_word_emb : numpy.ndarray, optional
        The pretrained word embeddings, default: ``None``.
    fix_word_emb : boolean, optional
        Specify whether to fix pretrained word embeddings, default: ``True``.

    Examples
    ----------
    >>> word_emb_layer = WordEmbedding(1000, 300, padding_idx=0, pretrained_word_emb=None, fix_word_emb=True)
    """
    def __init__(self, vocab_size, emb_size, padding_idx=0,
                    pretrained_word_emb=None, fix_word_emb=True):
        super(WordEmbedding, self).__init__()
        self.word_emb_layer = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx,
                            _weight=torch.from_numpy(pretrained_word_emb).float()
                            if pretrained_word_emb is not None else None)

        if fix_word_emb:
            print('[ Fix word embeddings ]')
            for param in self.word_emb_layer.parameters():
                param.requires_grad = False

    def forward(self, input_tensor):
        """Compute word embeddings.

        Parameters
        ----------
        input_tensor : torch.LongTensor
            The input word index sequence, shape: [num_items, max_size].

        Returns
        -------
        torch.Tensor
            Word embedding matrix.
        """
        return self.word_emb_layer(input_tensor)

class BertEmbedding(nn.Module):
    """Bert embedding class.
    """
    def __init__(self, fix_word_emb):
        super(BertEmbedding, self).__init__()
        self.fix_word_emb = fix_word_emb

    def forward(self):
        raise NotImplementedError()

class MeanEmbedding(nn.Module):
    """Mean embedding class.
    """
    def __init__(self):
        super(MeanEmbedding, self).__init__()

    def forward(self, emb, len_):
        """Compute average embeddings.

        Parameters
        ----------
        emb : torch.Tensor
            The input embedding tensor.
        len_ : torch.Tensor
            The sequence length tensor.

        Returns
        -------
        torch.Tensor
            The average embedding tensor.
        """
        return torch.sum(emb, dim=-2) / len_


class RNNEmbedding(nn.Module):
    """RNN embedding class: apply the RNN network to a sequence of word embeddings.

    Parameters
    ----------
    input_size : int
        The input feature size.
    hidden_size : int
        The hidden layer size.
    dropout : float, optional
        Dropout ratio, default: ``None``.
    bidirectional : boolean, optional
        Whether to use bidirectional RNN, default: ``False``.
    rnn_type : str
        The RNN cell type, default: ``lstm``.
    device : torch.device, optional
        Specify computation device (e.g., CPU), default: ``None`` for using CPU.
    """
    def __init__(self, input_size, hidden_size,
                    dropout=None, bidirectional=False,
                    rnn_type='lstm', device=None):
        super(RNNEmbedding, self).__init__()
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
        self.device = device
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(input_size, self.hidden_size, 1, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """Apply the RNN network to a sequence of word embeddings.

        Parameters
        ----------
        x : torch.Tensor
            The word embedding sequence.
        x_len : torch.LongTensor
            The input sequence length.

        Returns
        -------
        torch.Tensor
            The hidden states at every time step.
        torch.Tensor
            The hidden state at the last time step.
        """
        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)

        h0 = to_cuda(torch.zeros(self.num_directions, x_len.size(0), self.hidden_size), self.device)
        if self.rnn_type == 'lstm':
            c0 = to_cuda(torch.zeros(self.num_directions, x_len.size(0), self.hidden_size), self.device)
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
