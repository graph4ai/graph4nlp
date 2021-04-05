import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import dgl

from ..utils.generic_utils import to_cuda, dropout_fn, create_mask
from ..utils.bert_utils import *


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
    single_token_item : bool
        Specify whether the item (i.e., node or edge) contains single token or multiple tokens.
    emb_strategy : str
        Specify the embedding construction strategy including the following options:
            - 'w2v': use word2vec embeddings.
            - 'w2v_bilstm': use word2vec embeddings, and apply BiLSTM encoders.
            - 'w2v_bigru': use word2vec embeddings, and apply BiGRU encoders.
            - 'bert': use BERT embeddings.
            - 'bert_bilstm': use BERT embeddings, and apply BiLSTM encoders.
            - 'bert_bigru': use BERT embeddings, and apply BiGRU encoders.
            - 'w2v_bert': use word2vec and BERT embeddings.
            - 'w2v_bert_bilstm': use word2vec and BERT embeddings, and apply BiLSTM encoders.
            - 'w2v_bert_bigru': use word2vec and BERT embeddings, and apply BiGRU encoders.
    hidden_size : int, optional
        The hidden size of RNN layer, default: ``None``.
    num_rnn_layers : int, optional
        The number of RNN layers, default: ``1``.
    fix_word_emb : boolean, optional
        Specify whether to fix pretrained word embeddings, default: ``True``.
    fix_bert_emb : boolean, optional
        Specify whether to fix pretrained BERT embeddings, default: ``True``.
    bert_model_name : str, optional
        Specify the BERT model name, default: ``'bert-base-uncased'``.
    bert_lower_case : bool, optional
        Specify whether to lower case the input text for BERT embeddings, default: ``True``.
    word_dropout : float, optional
        Dropout ratio for word embedding, default: ``None``.
    rnn_dropout : float, optional
        Dropout ratio for RNN embedding, default: ``None``.
    device : torch.device, optional
        Specify computation device (e.g., CPU), default: ``None`` for using CPU.

    Note
    ----------
        word_emb_type : str or list of str
            Specify pretrained word embedding types including "w2v", "node_edge_bert", or "seq_bert".
        node_edge_emb_strategy : str
            Specify node/edge embedding strategies including "mean", "bilstm" and "bigru".
        seq_info_encode_strategy : str
            Specify strategies of encoding sequential information in raw text
            data including "none", "bilstm" and "bigru". You might
            want to do this in some situations, e.g., when all the nodes are single
            tokens extracted from the raw text.

        1) single-token node (i.e., single_token_item=`True`):
            a) 'w2v', 'bert', 'w2v_bert'
            b) node_edge_emb_strategy: 'mean'
            c) seq_info_encode_strategy: 'none', 'bilstm', 'bigru'
            emb_strategy: 'w2v', 'w2v_bilstm', 'w2v_bigru',
            'bert', 'bert_bilstm', 'bert_bigru',
            'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru'

        2) multi-token node (i.e., single_token_item=`False`):
            a) 'w2v', 'bert', 'w2v_bert'
            b) node_edge_emb_strategy: 'mean', 'bilstm', 'bigru'
            c) seq_info_encode_strategy: 'none'
            emb_strategy: ('w2v', 'w2v_bilstm', 'w2v_bigru',
            'bert', 'bert_bilstm', 'bert_bigru',
            'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru')
    """
    def __init__(self,
                    word_vocab,
                    single_token_item,
                    emb_strategy='w2v_bilstm',
                    hidden_size=None,
                    num_rnn_layers=1,
                    fix_word_emb=True,
                    fix_bert_emb=True,
                    bert_model_name='bert-base-uncased',
                    bert_lower_case=True,
                    word_dropout=None,
                    rnn_dropout=None,
                    device=None):
        super(EmbeddingConstruction, self).__init__()
        self.device = device
        self.word_dropout = word_dropout
        self.rnn_dropout = rnn_dropout

        assert emb_strategy in ('w2v', 'w2v_bilstm', 'w2v_bigru',
                        'bert', 'bert_bilstm', 'bert_bigru',
                        'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru'),\
            "emb_strategy must be one of ('w2v', 'w2v_bilstm', 'w2v_bigru', 'bert', 'bert_bilstm', 'bert_bigru', 'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru')"

        word_emb_type = set()
        if single_token_item:
            node_edge_emb_strategy = 'mean'
            if 'w2v' in emb_strategy:
                word_emb_type.add('w2v')

            if 'bert' in emb_strategy:
                word_emb_type.add('seq_bert')

            if 'bilstm' in emb_strategy:
                seq_info_encode_strategy = 'bilstm'
            elif 'bigru' in emb_strategy:
                seq_info_encode_strategy = 'bigru'
            else:
                seq_info_encode_strategy = 'none'
        else:
            seq_info_encode_strategy = 'none'
            if 'w2v' in emb_strategy:
                word_emb_type.add('w2v')

            if 'bert' in emb_strategy:
                word_emb_type.add('node_edge_bert')

            if 'bilstm' in emb_strategy:
                node_edge_emb_strategy = 'bilstm'
            elif 'bigru' in emb_strategy:
                node_edge_emb_strategy = 'bigru'
            else:
                node_edge_emb_strategy = 'mean'


        word_emb_size = 0
        self.word_emb_layers = nn.ModuleDict()
        if 'w2v' in word_emb_type:
            self.word_emb_layers['w2v'] = WordEmbedding(
                            word_vocab.embeddings.shape[0],
                            word_vocab.embeddings.shape[1],
                            pretrained_word_emb=word_vocab.embeddings,
                            fix_emb=fix_word_emb,
                            device=self.device)
            word_emb_size += word_vocab.embeddings.shape[1]

        if 'node_edge_bert' in word_emb_type:
            self.word_emb_layers['node_edge_bert'] = BertEmbedding(name=bert_model_name,
                                                            fix_emb=fix_bert_emb,
                                                            lower_case=bert_lower_case)
            word_emb_size += self.word_emb_layers['node_edge_bert'].bert_model.config.hidden_size

        if 'seq_bert' in word_emb_type:
            self.word_emb_layers['seq_bert'] = BertEmbedding(name=bert_model_name,
                                                            fix_emb=fix_bert_emb,
                                                            lower_case=bert_lower_case)

        if node_edge_emb_strategy in ('bilstm', 'bigru'):
            self.node_edge_emb_layer = RNNEmbedding(
                                    word_emb_size,
                                    hidden_size,
                                    bidirectional=True,
                                    num_layers=num_rnn_layers,
                                    rnn_type='lstm' if node_edge_emb_strategy == 'bilstm' else 'gru',
                                    dropout=rnn_dropout,
                                    device=device)
            rnn_input_size = hidden_size
        else:
            self.node_edge_emb_layer = MeanEmbedding()
            rnn_input_size = word_emb_size

        if 'seq_bert' in word_emb_type:
            rnn_input_size += self.word_emb_layers['seq_bert'].bert_model.config.hidden_size

        if seq_info_encode_strategy in ('bilstm', 'bigru'):
            self.output_size = hidden_size
            self.seq_info_encode_layer = RNNEmbedding(
                                    rnn_input_size,
                                    hidden_size,
                                    bidirectional=True,
                                    num_layers=num_rnn_layers,
                                    rnn_type='lstm' if seq_info_encode_strategy == 'bilstm' else 'gru',
                                    dropout=rnn_dropout,
                                    device=device)

            # apply a linear projection to make rnn_input_size equal hidden_size
            if rnn_input_size != hidden_size:
                self.linear_transform = nn.Linear(rnn_input_size, hidden_size, bias=False)

        else:
            self.output_size = rnn_input_size
            self.seq_info_encode_layer = None


    def forward(self, batch_gd, item_size, num_items, num_word_items=None):
        """Compute initial node/edge embeddings.

        Parameters
        ----------
        batch_gd : GraphData
            The batched graph data.
        item_size : torch.LongTensor
            The length of word sequence per item with shape :math:`(N)`.
        num_items : torch.LongTensor
            The number of items per graph with shape :math:`(B,)`
            where :math:`B` is the number of graphs in the batched graph.
        num_word_items : torch.LongTensor, optional
            The number of word items (that are extracted from the raw text)
            per graph with shape :math:`(B,)` where :math:`B` is the number
            of graphs in the batched graph. We assume that the word items are
            not reordered and interpolated, and always appear before the non-word
            items in the graph. Default: ``None``.

        Returns
        -------
        torch.Tensor
            The output item embeddings.
        """

        feat = []
        if 'w2v' in self.word_emb_layers:
            input_data = batch_gd.node_features['token_id'].long()
            feat.append(self.word_emb_layers['w2v'](input_data))

        if 'node_edge_bert' in self.word_emb_layers:
            input_data = [[batch_gd.node_attributes[i]['token']] for i in range(batch_gd.get_node_num())]
            feat.append(self.word_emb_layers['node_edge_bert'](input_data))

        if len(feat) > 0:
            feat = torch.cat(feat, dim=-1)
            feat = dropout_fn(feat, self.word_dropout, shared_axes=[-2], training=self.training)
            feat = self.node_edge_emb_layer(feat, item_size)
            if isinstance(feat, (tuple, list)):
                feat = feat[-1]


        if self.seq_info_encode_layer is None and 'seq_bert' not in self.word_emb_layers:
            return feat
        else:
            # unbatching
            new_feat = []
            raw_text_data = []
            start_idx = 0
            max_num_items = torch.max(num_items).item()
            for i in range(num_items.shape[0]):
                if len(feat) > 0:
                    tmp_feat = feat[start_idx: start_idx + num_items[i].item()]
                    if num_items[i].item() < max_num_items:
                        tmp_feat = torch.cat([tmp_feat, to_cuda(torch.zeros(
                            max_num_items - num_items[i], tmp_feat.shape[1]), self.device)], 0)
                    new_feat.append(tmp_feat)

                if 'seq_bert' in self.word_emb_layers:
                    raw_text_data.append([batch_gd.node_attributes[j]['token'] for j in range(start_idx, start_idx + num_items[i].item())])

                start_idx += num_items[i].item()

            # computation
            if len(new_feat) > 0:
                new_feat = torch.stack(new_feat, 0)

            if 'seq_bert' in self.word_emb_layers:
                bert_feat = self.word_emb_layers['seq_bert'](raw_text_data)
                if len(new_feat) > 0:
                    new_feat = torch.cat([new_feat, bert_feat], -1)
                else:
                    new_feat = bert_feat


            if self.seq_info_encode_layer is None:
                return new_feat

            len_ = num_word_items if num_word_items is not None else num_items
            rnn_state = self.seq_info_encode_layer(new_feat, len_)
            if isinstance(rnn_state, (tuple, list)):
                rnn_state = rnn_state[0]

            # batching
            ret_feat = []
            for i in range(len_.shape[0]):
                tmp_feat = rnn_state[i][:len_[i]]
                if len(tmp_feat) < num_items[i].item():
                    prev_feat = new_feat[i, len_[i]: num_items[i]]
                    if prev_feat.shape[-1] != tmp_feat.shape[-1]:
                        prev_feat = self.linear_transform(prev_feat)

                    tmp_feat = torch.cat([tmp_feat, prev_feat], 0)
                ret_feat.append(tmp_feat)

            ret_feat = torch.cat(ret_feat, 0)

            return ret_feat


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
    fix_emb : boolean, optional
        Specify whether to fix pretrained word embeddings, default: ``True``.

    Examples
    ----------
    >>> word_emb_layer = WordEmbedding(1000, 300, padding_idx=0, pretrained_word_emb=None, fix_emb=True)
    """
    def __init__(self, vocab_size, emb_size, padding_idx=0,
                    pretrained_word_emb=None, fix_emb=True, device=None):
        super(WordEmbedding, self).__init__()
        self.word_emb_layer = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx,
                            _weight=torch.from_numpy(pretrained_word_emb).float()
                            if pretrained_word_emb is not None else None)
        self.device = device

        if fix_emb:
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

    Parameters
    ----------
    name : str, optional
        BERT model name, default: ``'bert-base-uncased'``.
    max_seq_len : int, optional
        Maximal sequence length, default: ``500``.
    doc_stride : int, optional
        Chunking stride, default: ``250``.
    fix_emb : boolean, optional
        Specify whether to fix pretrained BERT embeddings, default: ``True``.
    lower_case : boolean, optional
        Specify whether to use lower case, default: ``True``.

    """
    def __init__(self,
                name='bert-base-uncased',
                max_seq_len=500,
                doc_stride=250,
                fix_emb=True,
                lower_case=True):
        super(BertEmbedding, self).__init__()
        self.bert_max_seq_len = max_seq_len
        self.bert_doc_stride = doc_stride
        self.fix_emb = fix_emb

        from transformers import BertModel
        from transformers import BertTokenizer
        print('[ Using pretrained BERT embeddings ]')
        self.bert_tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=lower_case)
        self.bert_model = BertModel.from_pretrained(name)
        if fix_emb:
            print('[ Fix BERT layers ]')
            self.bert_model.eval()
            for param in self.bert_model.parameters():
                param.requires_grad = False
        else:
            print('[ Finetune BERT layers ]')
            self.bert_model.train()

        # compute weighted average over BERT layers
        self.logits_bert_layers = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, self.bert_model.config.num_hidden_layers)))


    def forward(self, raw_text_data):
        """Compute BERT embeddings for each word in text.

        Parameters
        ----------
        raw_text_data : list
            The raw text input data. Example: [['what', 'is', 'bert'], ['how', 'to', 'use', 'bert']].

        Returns
        -------
        torch.Tensor
            BERT embedding matrix.
        """
        bert_features = []
        max_d_len = 0
        for text in raw_text_data:
            bert_features.append(convert_text_to_bert_features(text, self.bert_tokenizer, self.bert_max_seq_len, self.bert_doc_stride))
            max_d_len = max(max_d_len, len(text))

        max_bert_d_num_chunks = max([len(ex_bert_d) for ex_bert_d in bert_features])
        max_bert_d_len = max([len(bert_d.input_ids) for ex_bert_d in bert_features for bert_d in ex_bert_d])
        bert_xd = torch.LongTensor(len(raw_text_data), max_bert_d_num_chunks, max_bert_d_len).fill_(0)
        bert_xd_mask = torch.LongTensor(len(raw_text_data), max_bert_d_num_chunks, max_bert_d_len).fill_(0)
        for i, ex_bert_d in enumerate(bert_features): # Example level
            for j, bert_d in enumerate(ex_bert_d): # Chunk level
                bert_xd[i, j, :len(bert_d.input_ids)].copy_(torch.LongTensor(bert_d.input_ids))
                bert_xd_mask[i, j, :len(bert_d.input_mask)].copy_(torch.LongTensor(bert_d.input_mask))

        bert_xd = bert_xd.to(self.bert_model.device)
        bert_xd_mask = bert_xd_mask.to(self.bert_model.device)

        encoder_outputs = self.bert_model(bert_xd.view(-1, bert_xd.size(-1)),
                                        token_type_ids=None,
                                        attention_mask=bert_xd_mask.view(-1, bert_xd_mask.size(-1)),
                                        output_hidden_states=True,
                                        return_dict=True)
        all_encoder_layers = encoder_outputs['hidden_states'][1:] # The first one is the input embedding
        all_encoder_layers = torch.stack([x.view(bert_xd.shape + (-1,)) for x in all_encoder_layers], 0)
        bert_xd_f = extract_bert_hidden_states(all_encoder_layers, max_d_len, bert_features, weighted_avg=True)

        weights_bert_layers = torch.softmax(self.logits_bert_layers, dim=-1)
        bert_xd_f = torch.mm(weights_bert_layers, bert_xd_f.view(bert_xd_f.size(0), -1)).view(bert_xd_f.shape[1:])

        return bert_xd_f



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
        sumed_emb = torch.sum(emb, dim=1)
        len_ = len_.unsqueeze(1).expand_as(sumed_emb).float()
        return sumed_emb / len_


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
    def __init__(self,
                input_size,
                hidden_size,
                bidirectional=False,
                num_layers=1,
                rnn_type='lstm',
                dropout=None,
                device=None):
        super(RNNEmbedding, self).__init__()
        if not rnn_type in ('lstm', 'gru'):
            raise RuntimeError('rnn_type is expected to be lstm or gru, got {}'.format(rnn_type))

        # if bidirectional:
        #     print('[ Using bidirectional {} encoder ]'.format(rnn_type))
        # else:
        #     print('[ Using {} encoder ]'.format(rnn_type))

        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')

        self.device = device
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(input_size, self.hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

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

        h0 = to_cuda(torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size), self.device)
        if self.rnn_type == 'lstm':
            c0 = to_cuda(torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size), self.device)
            packed_h, (packed_h_t, _) = self.model(x, (h0, c0))
        else:
            packed_h, packed_h_t = self.model(x, h0)

        if self.num_layers > 1:
            # use the last RNN layer hidden state
            packed_h_t = packed_h_t.view(self.num_layers, self.num_directions, -1, self.hidden_size)[-1]

        if self.num_directions == 2:
            packed_h_t = torch.cat((packed_h_t[-1], packed_h_t[-2]), 1)
        else:
            packed_h_t = packed_h_t[-1]

        hh, _ = pad_packed_sequence(packed_h, batch_first=True)

        # restore the sorting
        _, inverse_indx = torch.sort(indx, 0)
        restore_hh = hh[inverse_indx]
        restore_packed_h_t = packed_h_t[inverse_indx]

        # add dropout
        restore_hh = dropout_fn(restore_hh, self.dropout, shared_axes=[-2], training=self.training)
        restore_packed_h_t = dropout_fn(restore_packed_h_t, self.dropout, training=self.training)

        return restore_hh, restore_packed_h_t
