import torch
import torch.nn as nn

from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import (
    BertEmbedding,
    EmbeddingConstructionBase,
    RNNEmbedding,
    WordEmbedding,
)
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.generic_utils import dropout_fn


class FusedEmbeddingConstruction(EmbeddingConstructionBase):
    """Initial graph embedding construction class.

    Parameters
    ----------
    word_vocab : Vocab
        The word vocabulary.
    single_token_item : bool
        Specify whether the item (i.e., node or edge) contains single token or multiple tokens.
    emb_strategy : str
        Specify the embedding construction strategy including the following options:
            - 'w2v_bilstm': use word2vec embeddings, and apply BiLSTM encoders.
            - 'w2v_bigru': use word2vec embeddings, and apply BiGRU encoders.
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
    bert_dropout : float, optional
        Dropout ratio for BERT embedding, default: ``None``.
    rnn_dropout : float, optional
        Dropout ratio for RNN embedding, default: ``None``.
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
        super(FusedEmbeddingConstruction, self).__init__()
        self.word_dropout = word_dropout
        self.bert_dropout = bert_dropout
        self.rnn_dropout = rnn_dropout

        assert emb_strategy in (
            "w2v",
            "w2v_bilstm",
            "w2v_bigru",
            "w2v_bert_bilstm",
            "w2v_bert_bigru",
        ), (
            "emb_strategy must be one of "
            "('w2v', 'w2v_bilstm', 'w2v_bigru', 'w2v_bert_bilstm', 'w2v_bert_bigru')"
        )

        word_emb_type = set()
        word_emb_type.add("w2v")

        if "bert" in emb_strategy:
            word_emb_type.add("seq_bert")

        if "bilstm" in emb_strategy:
            seq_info_encode_strategy = "bilstm"
        elif "bigru" in emb_strategy:
            seq_info_encode_strategy = "bigru"
        else:
            seq_info_encode_strategy = "none"

        word_emb_size = word_vocab.embeddings.shape[1]
        answer_feat_size = word_emb_size

        self.word_emb_layers = nn.ModuleDict()
        self.word_emb_layers["w2v"] = WordEmbedding(
            word_vocab.embeddings.shape[0],
            word_vocab.embeddings.shape[1],
            pretrained_word_emb=word_vocab.embeddings,
            fix_emb=fix_word_emb,
        )

        enc_input_dim = word_emb_size * 2
        bert_dim = 0
        if "seq_bert" in word_emb_type:
            self.word_emb_layers["seq_bert"] = BertEmbedding(
                name=bert_model_name, fix_emb=fix_bert_emb, lower_case=bert_lower_case
            )
            bert_dim = self.word_emb_layers["seq_bert"].bert_model.config.hidden_size
            answer_feat_size += bert_dim
            enc_input_dim += bert_dim

        self.ans_rnn_encoder = RNNEmbedding(
            answer_feat_size,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=1,
            rnn_type="lstm" if seq_info_encode_strategy == "bilstm" else "gru",
            dropout=rnn_dropout,
        )

        self.ctx_rnn_encoder = RNNEmbedding(
            enc_input_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=1,
            rnn_type="lstm" if seq_info_encode_strategy == "bilstm" else "gru",
            dropout=rnn_dropout,
        )

        self.ctx_rnn_encoder_l2 = RNNEmbedding(
            2 * hidden_size,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=1,
            rnn_type="lstm" if seq_info_encode_strategy == "bilstm" else "gru",
            dropout=rnn_dropout,
        )

        self.ctx2ans_attn_l1 = Context2AnswerAttention(word_emb_size, hidden_size)
        self.ctx2ans_attn_l2 = Context2AnswerAttention(
            word_emb_size + hidden_size + bert_dim, hidden_size
        )

    def forward(self, data):
        """Compute initial node/edge embeddings.

        Parameters
        ----------
        data

        Returns
        -------
        torch.Tensor
            The output item embeddings.
        """
        batch_gd = data["graph_data"]
        assert (
            batch_gd.batch_node_features["token_id"].shape[2] == 1
        ), "must be single-token node graph"

        ctx_length = torch.LongTensor(batch_gd._batch_num_nodes).to(batch_gd.device)

        # passage encoding
        token_ids = batch_gd.batch_node_features["token_id"].squeeze(2)
        encoder_embedded = self.word_emb_layers["w2v"](token_ids)
        encoder_embedded = dropout_fn(
            encoder_embedded, self.word_dropout, shared_axes=[-2], training=self.training
        )
        enc_input_cat = [encoder_embedded]

        # answer encoding
        ans_embedded = self.word_emb_layers["w2v"](data["input_tensor2"])
        ans_embedded = dropout_fn(
            ans_embedded, self.word_dropout, shared_axes=[-2], training=self.training
        )
        answer_feat = ans_embedded

        # Align answer info to passage at the word level
        ans_mask = data["input_tensor2"] != 0
        ctx_aware_ans_emb = self.ctx2ans_attn_l1(
            encoder_embedded, ans_embedded, ans_embedded, ans_mask
        )
        enc_input_cat.append(ctx_aware_ans_emb)

        if "seq_bert" in self.word_emb_layers:
            # passage encoding
            gd_list = from_batch(batch_gd)
            raw_tokens = [
                [gd.node_attributes[i]["token"] for i in range(gd.get_node_num())] for gd in gd_list
            ]
            ctx_bert = self.word_emb_layers["seq_bert"](raw_tokens)
            ctx_bert = dropout_fn(
                ctx_bert, self.bert_dropout, shared_axes=[-2], training=self.training
            )
            enc_input_cat.append(ctx_bert)

            # answer encoding
            answer_raw_tokens = data["input_text2"]
            answer_bert = self.word_emb_layers["seq_bert"](answer_raw_tokens)
            answer_bert = dropout_fn(
                answer_bert, self.bert_dropout, shared_axes=[-2], training=self.training
            )
            answer_feat = torch.cat([answer_feat, answer_bert], -1)

        enc_input_cat = torch.cat(enc_input_cat, -1)
        ctx_feat = self.ctx_rnn_encoder(enc_input_cat, ctx_length)
        ctx_feat = ctx_feat[0] if isinstance(ctx_feat, (tuple, list)) else ctx_feat
        answer_feat = self.ans_rnn_encoder(answer_feat, data["input_length2"])
        answer_feat = answer_feat[0] if isinstance(answer_feat, (tuple, list)) else answer_feat

        # Align answer info to passage at the word level
        enc_cat_l2 = torch.cat([encoder_embedded, ctx_feat], -1)
        ans_cat_l2 = torch.cat([ans_embedded, answer_feat], -1)

        if "seq_bert" in self.word_emb_layers:
            enc_cat_l2 = torch.cat([enc_cat_l2, ctx_bert], -1)
            ans_cat_l2 = torch.cat([ans_cat_l2, answer_bert], -1)

        ctx_aware_ans_emb = self.ctx2ans_attn_l2(enc_cat_l2, ans_cat_l2, answer_feat, ans_mask)
        ctx_feat = self.ctx_rnn_encoder_l2(torch.cat([ctx_feat, ctx_aware_ans_emb], -1), ctx_length)
        ctx_feat = ctx_feat[0] if isinstance(ctx_feat, (tuple, list)) else ctx_feat
        batch_gd.batch_node_features["node_feat"] = ctx_feat

        return batch_gd


class Context2AnswerAttention(nn.Module):
    def __init__(self, dim, hidden_size):
        super(Context2AnswerAttention, self).__init__()
        self.linear_sim = nn.Linear(dim, hidden_size, bias=False)

    def forward(self, context, answers, out_answers, mask_answers=None):
        """
        Parameters
        :context, (B, L, dim)
        :answers, (B, N, dim)
        :mask, (L, N)

        Returns
        :ques_emb, (L, dim)
        """
        context_fc = torch.relu(self.linear_sim(context))
        questions_fc = torch.relu(self.linear_sim(answers))

        # shape: (B, L, N)
        attention = torch.matmul(context_fc, questions_fc.transpose(-1, -2))

        if mask_answers is not None:
            attention = attention.masked_fill(~mask_answers.unsqueeze(1).bool(), -Constants.INF)

        prob = torch.softmax(attention, dim=-1)
        # shape: (B, L, dim)
        emb = torch.matmul(prob, out_answers)

        return emb
