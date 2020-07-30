import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RNNTreeDecoderBase
from ...utils.tree_utils import Tree, to_cuda


class StdTreeDecoder(RNNTreeDecoderBase):
    def __init__(self, attn, dec_emb_size, dec_hidden_size, output_size, device, criterion, use_sibling = True, batch_size=20, use_attention=True, use_copy=False, use_coverage=False,
                 fuse_strategy="average", num_layers=1, dropout_input=0.1, dropout_output=0.3, rnn_type="lstm", max_dec_seq_length=512, max_dec_tree_depth=256, tgt_vocab=None):
        super(StdTreeDecoder, self).__init__(use_attention=True, use_copy=False, use_coverage=False, attention_type="uniform",
                                             fuse_strategy="average")
        self.num_layers = num_layers
        self.device = device
        self.criterion = criterion
        self.batch_size = batch_size
        self.rnn_size = dec_hidden_size
        self.hidden_size = dec_hidden_size
        self.max_dec_seq_length = max_dec_seq_length
        self.max_dec_tree_depth = max_dec_tree_depth
        self.tgt_vocab = tgt_vocab

                
        self.attn_state = {}

        self.rnn = self._build_rnn(
            rnn_type, input_size=output_size, emb_size=dec_emb_size, hidden_size=dec_hidden_size, dropout_input=dropout_input, dropout_output=dropout_output)
        self.coverage_attention = use_coverage
        self.copy_attention = use_copy
        self.attention = attn

    def _run_forward_pass(self, graph_node_embedding, graph_node_mask, rnn_node_embedding, graph_level_embedding,
                          graph_edge_embedding=None, graph_edge_mask=None, tgt_tree_batch=None):
        r"""
            The private calculation method for decoder.

        Parameters
        ----------
        tgt_seq: torch.Tensor,
            The target sequence of shape :math:`(B, seq_len)`, where :math:`B`
            is size of batch, :math:`seq_len` is the length of sequences.
            Note that it is consisted by tokens' index.
        graph_node_embedding: torch.Tensor,
            The graph node embedding matrix of shape :math:`(B, N, D_{in})`
        graph_node_mask: torch.Tensor,
            The graph node type mask matrix of shape :math`(B, N)`
        rnn_node_embedding: torch.Tensor,
            The rnn encoded embedding matrix of shape :math`(B, N, D_{in})`
        graph_level_embedding: torch.Tensor,
            graph level embedding of shape :math`(B, D_{in})`
        graph_edge_embedding: torch.Tensor,
            graph edge embedding of shape :math`(B, N, D_{in})`
        graph_edge_mask: torch.Tensor,
            graph edge type embedding

        Returns
        ----------
        logits: torch.Tensor
        attns: torch.Tensor
        """

        tgt_batch_size = len(tgt_tree_batch)
        # assert graph_node_embedding.requires_grad == True
        assert rnn_node_embedding.requires_grad == True
        assert graph_level_embedding.requires_grad == True

        enc_outputs = rnn_node_embedding
        graph_cell_state = graph_level_embedding
        graph_hidden_state = graph_level_embedding

        cur_index = 1
        loss = 0

        dec_batch, queue_tree, max_index = get_dec_batch(
            tgt_tree_batch, tgt_batch_size, False, self.tgt_vocab)


        dec_state = {}
        for i in range(self.max_dec_tree_depth + 1):
            dec_state[i] = {}
            for j in range(self.max_dec_seq_length + 1):
                dec_state[i][j] = {}

        while (cur_index <= max_index):
            for j in range(1, 3):
                dec_state[cur_index][0][j] = torch.zeros(
                    (self.batch_size, self.rnn_size), dtype=torch.float, requires_grad=True)
                to_cuda(dec_state[cur_index][0][j], self.device)

            # sibling_state = torch.zeros(
            #     (self.batch_size, self.rnn_size), dtype=torch.float, requires_grad=True)
            # to_cuda(sibling_state, self.device)

            if cur_index == 1:
                for i in range(self.batch_size):
                    # print(dec_state[1][0][1].is_leaf)
                    dec_state[1][0][1][i, :] = graph_cell_state[i]
                    dec_state[1][0][2][i, :] = graph_hidden_state[i]

            else:
                for i in range(1, self.batch_size+1):
                    if (cur_index <= len(queue_tree[i])):
                        par_index = queue_tree[i][cur_index - 1]["parent"]
                        child_index = queue_tree[i][cur_index -
                                                    1]["child_index"]

                        dec_state[cur_index][0][1][i-1, :] = \
                            dec_state[par_index][child_index][1][i-1, :]
                        dec_state[cur_index][0][2][i-1,
                                                        :] = dec_state[par_index][child_index][2][i-1, :]

                    # flag_sibling = False
                    # for q_index in range(len(queue_tree[i])):
                    #     if (cur_index <= len(queue_tree[i])) and (q_index < cur_index - 1) and (queue_tree[i][q_index]["parent"] == queue_tree[i][cur_index - 1]["parent"]) and (queue_tree[i][q_index]["child_index"] < queue_tree[i][cur_index - 1]["child_index"]):
                    #         flag_sibling = True
                    #         # sibling_index = queue_tree[i][q_index]["child_index"]
                    #         sibling_index = q_index
                    # if flag_sibling:
                    #     sibling_state[i - 1, :] = dec_state[sibling_index][dec_batch[sibling_index].size(
                    #         1) - 1][2][i - 1, :]

            parent_h = dec_state[cur_index][0][2]
            for i in range(dec_batch[cur_index].size(1) - 1):
                dec_state[cur_index][i+1][1], dec_state[cur_index][i+1][2] = self.rnn(
                    dec_batch[cur_index][:, i], dec_state[cur_index][i][1], dec_state[cur_index][i][2], parent_h)
                # print(enc_outputs.is_leaf)
                pred = self.attention(
                    enc_outputs, dec_state[cur_index][i+1][2], torch.tensor(0))
                # output_tree_batch
                # print(pred.is_leaf)
                # print(i)
                loss += self.criterion(pred, dec_batch[cur_index][:, i+1])
                # print(loss.is_leaf)
            cur_index = cur_index + 1
        loss = loss / self.batch_size
        return loss

    def _build_rnn(self, rnn_type, input_size, emb_size, hidden_size, dropout_input, dropout_output):
        rnn = DecoderRNN(input_size, emb_size, hidden_size, dropout_input, dropout_output)
        return rnn


class Dec_LSTM(nn.Module):
    '''
    Decoder LSTM cell with parent hidden state
    '''

    def __init__(self, emb_size, hidden_size, dropout):
        super(Dec_LSTM, self).__init__()
        self.rnn_size = hidden_size
        self.word_embedding_size = emb_size
        self.i2h = nn.Linear(self.word_embedding_size +
                             self.rnn_size, 4*self.rnn_size)

        self.h2h = nn.Linear(self.rnn_size, 4*self.rnn_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev_c, prev_h, parent_h):
        input_cat = torch.cat((x, parent_h), 1)
        gates = self.i2h(input_cat) + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return cy, hy


class DecoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, dropout_input, dropout_output):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_embedding_size = emb_size
        self.embedding = nn.Embedding(
            input_size, self.word_embedding_size, padding_idx=0)

        self.lstm = Dec_LSTM(emb_size, hidden_size, dropout_output)
        self.dropout = nn.Dropout(dropout_input)

    def forward(self, input_src, prev_c, prev_h, parent_h):

        src_emb = self.embedding(input_src)
        src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(
            src_emb, prev_c, prev_h, parent_h)
        return prev_cy, prev_hy


def get_dec_batch(dec_tree_batch, batch_size, device, form_manager):
    queue_tree = {}
    for i in range(1, batch_size+1):
        queue_tree[i] = []
        queue_tree[i].append(
            {"tree": dec_tree_batch[i-1], "parent": 0, "child_index": 1})

    cur_index, max_index = 1, 1
    dec_batch = {}
    # max_index: the max number of sequence decoder in one batch
    while (cur_index <= max_index):
        max_w_len = -1
        batch_w_list = []
        for i in range(1, batch_size+1):
            w_list = []
            if (cur_index <= len(queue_tree[i])):
                t = queue_tree[i][cur_index - 1]["tree"]

                for ic in range(t.num_children):
                    if isinstance(t.children[ic], Tree):
                        w_list.append(4)
                        queue_tree[i].append(
                            {"tree": t.children[ic], "parent": cur_index, "child_index": ic + 1})
                    else:
                        w_list.append(t.children[ic])
                if len(queue_tree[i]) > max_index:
                    max_index = len(queue_tree[i])
            if len(w_list) > max_w_len:
                max_w_len = len(w_list)
            batch_w_list.append(w_list)
        dec_batch[cur_index] = torch.zeros(
            (batch_size, max_w_len + 2), dtype=torch.long)
        for i in range(batch_size):
            w_list = batch_w_list[i]
            if len(w_list) > 0:
                for j in range(len(w_list)):
                    dec_batch[cur_index][i][j+1] = w_list[j]
                # add <S>, <E>
                if cur_index == 1:
                    dec_batch[cur_index][i][0] = 1
                else:
                    dec_batch[cur_index][i][0] = form_manager.get_symbol_idx(
                        '(')
                dec_batch[cur_index][i][len(w_list) + 1] = 2

        to_cuda(dec_batch[cur_index], device)
        cur_index += 1

    return dec_batch, queue_tree, max_index
