import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ...data.data import from_batch
from ..utils.bert_utils import convert_text_to_bert_features, extract_bert_hidden_states
from ..utils.generic_utils import dropout_fn
from ..utils.padding_utils import pad_4d_vals
from ..utils.vocab_utils import Vocab


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
        Note that if 'w2v' is not applied, `pretrained_word_emb_name` specified in Dataset APIs
        will be superseded.
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

    Note
    ----------
        word_emb_type : str or list of str
            Specify pretrained word embedding types including "w2v", "node_edge_bert",
            or "seq_bert".
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

    def __init__(
        self,
        word_vocab,
        single_token_item,
        emb_strategy="w2v_bilstm",
        hidden_size=None,
        num_rnn_layers=1,
        fix_word_emb=True,
        fix_bert_emb=True,
        bert_model_name="bert-base-uncased",
        bert_lower_case=True,
        word_dropout=None,
        bert_dropout=None,
        rnn_dropout=None,
    ):
        super(EmbeddingConstruction, self).__init__()
        self.word_dropout = word_dropout
        self.bert_dropout = bert_dropout
        self.rnn_dropout = rnn_dropout
        self.single_token_item = single_token_item

        assert emb_strategy in (
            "w2v",
            "w2v_bilstm",
            "w2v_bigru",
            "bert",
            "bert_bilstm",
            "bert_bigru",
            "w2v_bert",
            "w2v_bert_bilstm",
            "w2v_bert_bigru",
        ), "emb_strategy must be one of ('w2v', 'w2v_bilstm', 'w2v_bigru', 'bert', 'bert_bilstm', "
        "'bert_bigru', 'w2v_bert', 'w2v_bert_bilstm', 'w2v_bert_bigru')"

        word_emb_type = set()
        if single_token_item:
            node_edge_emb_strategy = None
            if "w2v" in emb_strategy:
                word_emb_type.add("w2v")

            if "bert" in emb_strategy:
                word_emb_type.add("seq_bert")

            if "bilstm" in emb_strategy:
                seq_info_encode_strategy = "bilstm"
            elif "bigru" in emb_strategy:
                seq_info_encode_strategy = "bigru"
            else:
                seq_info_encode_strategy = "none"
        else:
            seq_info_encode_strategy = "none"
            if "w2v" in emb_strategy:
                word_emb_type.add("w2v")

            if "bert" in emb_strategy:
                word_emb_type.add("node_edge_bert")

            if "bilstm" in emb_strategy:
                node_edge_emb_strategy = "bilstm"
            elif "bigru" in emb_strategy:
                node_edge_emb_strategy = "bigru"
            else:
                node_edge_emb_strategy = "mean"

        word_emb_size = 0
        self.word_emb_layers = nn.ModuleDict()
        if "w2v" in word_emb_type:
            self.word_emb_layers["w2v"] = WordEmbedding(
                word_vocab.embeddings.shape[0],
                word_vocab.embeddings.shape[1],
                pretrained_word_emb=word_vocab.embeddings,
                fix_emb=fix_word_emb,
            )
            word_emb_size += word_vocab.embeddings.shape[1]

        if "node_edge_bert" in word_emb_type:
            self.word_emb_layers["node_edge_bert"] = BertEmbedding(
                name=bert_model_name, fix_emb=fix_bert_emb, lower_case=bert_lower_case
            )
            word_emb_size += self.word_emb_layers["node_edge_bert"].bert_model.config.hidden_size

        if "seq_bert" in word_emb_type:
            self.word_emb_layers["seq_bert"] = BertEmbedding(
                name=bert_model_name, fix_emb=fix_bert_emb, lower_case=bert_lower_case
            )

        if node_edge_emb_strategy in ("bilstm", "bigru"):
            self.node_edge_emb_layer = RNNEmbedding(
                word_emb_size,
                hidden_size,
                bidirectional=True,
                num_layers=num_rnn_layers,
                rnn_type="lstm" if node_edge_emb_strategy == "bilstm" else "gru",
                dropout=rnn_dropout,
            )
            rnn_input_size = hidden_size
        elif node_edge_emb_strategy == "mean":
            self.node_edge_emb_layer = MeanEmbedding()
            rnn_input_size = word_emb_size
        else:
            rnn_input_size = word_emb_size

        if "seq_bert" in word_emb_type:
            rnn_input_size += self.word_emb_layers["seq_bert"].bert_model.config.hidden_size

        if seq_info_encode_strategy in ("bilstm", "bigru"):
            self.output_size = hidden_size
            self.seq_info_encode_layer = RNNEmbedding(
                rnn_input_size,
                hidden_size,
                bidirectional=True,
                num_layers=num_rnn_layers,
                rnn_type="lstm" if seq_info_encode_strategy == "bilstm" else "gru",
                dropout=rnn_dropout,
            )

        else:
            self.output_size = rnn_input_size
            self.seq_info_encode_layer = None

    def forward(self, batch_gd):
        """Compute initial node/edge embeddings.

        Parameters
        ----------
        batch_gd : GraphData
            The input graph data.

        Returns
        -------
        GraphData
            The output graph data with updated node embeddings.
        """
        feat = []
        if self.single_token_item:  # single-token node graph
            token_ids = batch_gd.batch_node_features["token_id"]
            if "w2v" in self.word_emb_layers:
                word_feat = self.word_emb_layers["w2v"](token_ids).squeeze(-2)
                word_feat = dropout_fn(
                    word_feat, self.word_dropout, shared_axes=[-2], training=self.training
                )
                feat.append(word_feat)

        else:  # multi-token node graph
            token_ids = batch_gd.node_features["token_id"]
            if "w2v" in self.word_emb_layers:
                word_feat = self.word_emb_layers["w2v"](token_ids)
                word_feat = dropout_fn(
                    word_feat, self.word_dropout, shared_axes=[-2], training=self.training
                )
                feat.append(word_feat)

            if "node_edge_bert" in self.word_emb_layers:
                input_data = [
                    batch_gd.node_attributes[i]["token"].strip().split(" ")
                    for i in range(batch_gd.get_node_num())
                ]
                node_edge_bert_feat = self.word_emb_layers["node_edge_bert"](input_data)
                node_edge_bert_feat = dropout_fn(
                    node_edge_bert_feat, self.bert_dropout, shared_axes=[-2], training=self.training
                )
                feat.append(node_edge_bert_feat)

            if len(feat) > 0:
                feat = torch.cat(feat, dim=-1)
                node_token_lens = torch.clamp((token_ids != Vocab.PAD).sum(-1), min=1)
                feat = self.node_edge_emb_layer(feat, node_token_lens)
                if isinstance(feat, (tuple, list)):
                    feat = feat[-1]

                feat = batch_gd.split_features(feat)

        if self.seq_info_encode_layer is None and "seq_bert" not in self.word_emb_layers:
            if isinstance(feat, list):
                feat = torch.cat(feat, -1)

            batch_gd.batch_node_features["node_feat"] = feat

            return batch_gd
        else:  # single-token node graph
            new_feat = feat
            if "seq_bert" in self.word_emb_layers:
                gd_list = from_batch(batch_gd)
                raw_tokens = [
                    [gd.node_attributes[i]["token"] for i in range(gd.get_node_num())]
                    for gd in gd_list
                ]
                bert_feat = self.word_emb_layers["seq_bert"](raw_tokens)
                bert_feat = dropout_fn(
                    bert_feat, self.bert_dropout, shared_axes=[-2], training=self.training
                )
                new_feat.append(bert_feat)

            new_feat = torch.cat(new_feat, -1)
            if self.seq_info_encode_layer is None:
                batch_gd.batch_node_features["node_feat"] = new_feat

                return batch_gd

            rnn_state = self.seq_info_encode_layer(
                new_feat, torch.LongTensor(batch_gd._batch_num_nodes).to(batch_gd.device)
            )
            if isinstance(rnn_state, (tuple, list)):
                rnn_state = rnn_state[0]

            batch_gd.batch_node_features["node_feat"] = rnn_state

            return batch_gd


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
    >>> emb_layer = WordEmbedding(500, 300, padding_idx=0, pretrained_word_emb=None, fix_emb=True)
    """

    def __init__(self, vocab_size, emb_size, padding_idx=0, pretrained_word_emb=None, fix_emb=True):
        super(WordEmbedding, self).__init__()
        self.word_emb_layer = nn.Embedding(
            vocab_size,
            emb_size,
            padding_idx=padding_idx,
            _weight=torch.from_numpy(pretrained_word_emb).float()
            if pretrained_word_emb is not None
            else None,
        )

        if fix_emb:
            print("[ Fix word embeddings ]")
            for param in self.word_emb_layer.parameters():
                param.requires_grad = False

    @property
    def weight(self):
        return self.word_emb_layer.weight

    @property
    def embedding_dim(self):
        return self.word_emb_layer.embedding_dim

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

    def __init__(
        self,
        name="bert-base-uncased",
        max_seq_len=500,
        doc_stride=250,
        fix_emb=True,
        lower_case=True,
    ):
        super(BertEmbedding, self).__init__()
        self.bert_max_seq_len = max_seq_len
        self.bert_doc_stride = doc_stride
        self.fix_emb = fix_emb

        from transformers import BertModel
        from transformers import BertTokenizer

        print("[ Using pretrained BERT embeddings ]")
        self.bert_tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=lower_case)
        self.bert_model = BertModel.from_pretrained(name)
        if fix_emb:
            print("[ Fix BERT layers ]")
            self.bert_model.eval()
            for param in self.bert_model.parameters():
                param.requires_grad = False
        else:
            print("[ Finetune BERT layers ]")
            self.bert_model.train()

        # compute weighted average over BERT layers
        self.logits_bert_layers = nn.Parameter(
            nn.init.xavier_uniform_(torch.Tensor(1, self.bert_model.config.num_hidden_layers))
        )

    def forward(self, raw_text_data):
        """Compute BERT embeddings for each word in text.

        Parameters
        ----------
        raw_text_data : list
            The raw text input data. Example: [['what', 'is', 'bert'], ['how', 'to', 'use']].

        Returns
        -------
        torch.Tensor
            BERT embedding matrix.
        """
        bert_features = []
        max_d_len = 0
        for text in raw_text_data:
            bert_features.append(
                convert_text_to_bert_features(
                    text, self.bert_tokenizer, self.bert_max_seq_len, self.bert_doc_stride
                )
            )
            max_d_len = max(max_d_len, len(text))

        max_bert_d_num_chunks = max([len(ex_bert_d) for ex_bert_d in bert_features])
        max_bert_d_len = max(
            [len(bert_d.input_ids) for ex_bert_d in bert_features for bert_d in ex_bert_d]
        )
        bert_xd = torch.LongTensor(len(raw_text_data), max_bert_d_num_chunks, max_bert_d_len).fill_(
            0
        )
        bert_xd_mask = torch.LongTensor(
            len(raw_text_data), max_bert_d_num_chunks, max_bert_d_len
        ).fill_(0)
        bert_token_to_orig_map = []
        for i, ex_bert_d in enumerate(bert_features):  # Example level
            ex_token_to_orig_map = []
            for j, bert_d in enumerate(ex_bert_d):  # Chunk level
                bert_xd[i, j, : len(bert_d.input_ids)].copy_(torch.LongTensor(bert_d.input_ids))
                bert_xd_mask[i, j, : len(bert_d.input_mask)].copy_(
                    torch.LongTensor(bert_d.input_mask)
                )
                ex_token_to_orig_map.append(bert_d.token_to_orig_map_matrix)
            bert_token_to_orig_map.append(ex_token_to_orig_map)
        bert_token_to_orig_map = pad_4d_vals(
            bert_token_to_orig_map,
            len(raw_text_data),
            max_bert_d_num_chunks,
            max_bert_d_len,
            max_d_len,
        )
        bert_token_to_orig_map = torch.Tensor(bert_token_to_orig_map).to(self.bert_model.device)

        bert_xd = bert_xd.to(self.bert_model.device)
        bert_xd_mask = bert_xd_mask.to(self.bert_model.device)

        encoder_outputs = self.bert_model(
            bert_xd.view(-1, bert_xd.size(-1)),
            token_type_ids=None,
            attention_mask=bert_xd_mask.view(-1, bert_xd_mask.size(-1)),
            output_hidden_states=True,
            return_dict=True,
        )

        all_encoder_layers = encoder_outputs["hidden_states"][
            1:
        ]  # The first one is the input embedding
        all_encoder_layers = torch.stack(
            [x.view(bert_xd.shape + (-1,)) for x in all_encoder_layers], 0
        )
        bert_xd_f = extract_bert_hidden_states(
            all_encoder_layers, bert_token_to_orig_map, weighted_avg=True
        )

        weights_bert_layers = torch.softmax(self.logits_bert_layers, dim=-1)
        bert_xd_f = torch.mm(weights_bert_layers, bert_xd_f.view(bert_xd_f.size(0), -1)).view(
            bert_xd_f.shape[1:]
        )

        return bert_xd_f


class MeanEmbedding(nn.Module):
    """Mean embedding class."""

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
        summed = torch.sum(emb, dim=-2)
        len_ = len_.unsqueeze(-1).expand_as(summed).float()
        return summed / len_


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
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bidirectional=False,
        num_layers=1,
        rnn_type="lstm",
        dropout=None,
    ):
        super(RNNEmbedding, self).__init__()
        if rnn_type not in ("lstm", "gru"):
            raise RuntimeError("rnn_type is expected to be lstm or gru, got {}".format(rnn_type))

        # if bidirectional:
        #     print('[ Using bidirectional {} encoder ]'.format(rnn_type))
        # else:
        #     print('[ Using {} encoder ]'.format(rnn_type))

        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError("hidden_size is expected to be even in the bidirectional mode!")

        self.dropout = dropout
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        model = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.model = model(
            input_size, self.hidden_size, num_layers, batch_first=True, bidirectional=bidirectional
        )

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
        device = x.device
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)

        h0 = torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size).to(
            device
        )
        if self.rnn_type == "lstm":
            c0 = torch.zeros(
                self.num_directions * self.num_layers, x_len.size(0), self.hidden_size
            ).to(device)
            packed_h, (packed_h_t, _) = self.model(x, (h0, c0))
        else:
            packed_h, packed_h_t = self.model(x, h0)

        if self.num_layers > 1:
            # use the last RNN layer hidden state
            packed_h_t = packed_h_t.view(
                self.num_layers, self.num_directions, -1, self.hidden_size
            )[-1]

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
