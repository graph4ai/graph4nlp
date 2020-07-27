import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RNNTreeDecoderBase

class StdTreeDecoder(RNNTreeDecoderBase):
    def __init__(self, attentional=True, use_copy=False, use_coverage=False, attention_type="uniform",
                 fuse_strategy="average", teacher_forcing_ratio=1., embeddings=None, hidden_size=300, num_layers=1, dropout=0.3, rnn_type="lstm", max_dec_seq_length=512, max_dec_tree_depth=256, tgt_vocab=None):
        super(StdTreeDecoder, self).__init__(attentional=True, use_copy=False, use_coverage=False, attention_type="uniform",
                                               fuse_strategy="average", teacher_forcing_ratio=1.)
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.max_dec_seq_length = max_dec_seq_length
        self.max_dec_tree_depth = max_dec_tree_depth
        self.tgt_vocab = tgt_vocab

        self.dec_state = {}
        self.attn_state = {}

        self.rnn = self._build_rnn(
            rnn_type, input_size=embeddings.vocabulary_size, hidden_size=hidden_size, dropout=dropout)
        self.coverage_attention = use_coverage
        self.copy_attention = use_copy
        self.attention = AttnUnit(hidden_size, embeddings.vocabulary_size, "separate_different_encoder_type", dropout)

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

        assert graph_node_embedding.requires_grad == True
        assert rnn_node_embedding.requires_grad == True
        assert graph_level_embedding.requires_grad == True
        
        enc_outputs = graph_node_embedding
        graph_cell_state = graph_level_embedding
        graph_hidden_state = graph_level_embedding

        for i in range(self.max_dec_tree_depth + 1):
            self.dec_state[i] = {}
            for j in range(self.max_dec_seq_length + 1):
                self.dec_state[i][j] = {}
        
        cur_index = 1

        dec_batch, queue_tree, max_index = get_dec_batch(tgt_tree_batch, tgt_batch_size, self.tgt_vocab, using_gpu=False)

        while (cur_index <= max_index):
            for j in range(1, 3):
                self.dec_state[cur_index][0][j] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
                if using_gpu:
                    self.dec_state[cur_index][0][j] = self.dec_state[cur_index][0][j].cuda()

            sibling_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
            if using_gpu:
                    sibling_state = sibling_state.cuda()

            if cur_index == 1:
                for i in range(opt.batch_size):
                    self.dec_state[1][0][1][i, :] = graph_cell_state[i]
                    self.dec_state[1][0][2][i, :] = graph_hidden_state[i]

            else:
                for i in range(1, opt.batch_size+1):
                    if (cur_index <= len(queue_tree[i])):
                        par_index = queue_tree[i][cur_index - 1]["parent"]
                        child_index = queue_tree[i][cur_index - 1]["child_index"]

                        self.dec_state[cur_index][0][1][i-1,:] = \
                            self.dec_state[par_index][child_index][1][i-1,:]
                        self.dec_state[cur_index][0][2][i-1,:] = self.dec_state[par_index][child_index][2][i-1,:]

                    flag_sibling = False
                    for q_index in range(len(queue_tree[i])):
                        if (cur_index <= len(queue_tree[i])) and (q_index < cur_index - 1) and (queue_tree[i][q_index]["parent"] == queue_tree[i][cur_index - 1]["parent"]) and (queue_tree[i][q_index]["child_index"] < queue_tree[i][cur_index - 1]["child_index"]):
                            flag_sibling = True
                            # sibling_index = queue_tree[i][q_index]["child_index"]
                            sibling_index = q_index
                    if flag_sibling:
                        sibling_state[i - 1, :] = self.dec_state[sibling_index][dec_batch[sibling_index].size(1) - 1][2][i - 1,:]

            parent_h = self.dec_state[cur_index][0][2]
            for i in range(dec_batch[cur_index].size(1) - 1):
                self.dec_state[cur_index][i+1][1], self.dec_state[cur_index][i+1][2] = decoder(dec_batch[cur_index][:,i], self.dec_state[cur_index][i][1], self.dec_state[cur_index][i][2], parent_h, sibling_state)
                pred = attention_decoder(enc_outputs, self.dec_state[cur_index][i+1][2], structural_info)

                loss += criterion(pred, dec_batch[cur_index][:,i+1])
            cur_index = cur_index + 1


    def _build_rnn(self, rnn_type, input_size, hidden_size, dropout):
        rnn = DecoderRNN(input_size, hidden_size, dropout)
        return rnn

class AttnUnit(nn.Module):
    def __init__(self, hidden_size, output_size, attention_type, dropout):
        super(AttnUnit, self).__init__()
        self.hidden_size = hidden_size
        self.separate_attention = (attention_type!=None)
        
        if self.separate_attention == "separate_different_encoder_type":
            self.linear_att = nn.Linear(3*self.hidden_size, self.hidden_size)
        else:
            self.linear_att = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.linear_out = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top, enc_2):
        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        enc_attention = torch.bmm(enc_s_top.permute(0, 2, 1), attention)

        if self.separate_attention == "separate_different_encoder_type":
            dot_2 = torch.bmm(enc_2, dec_s_top.unsqueeze(2))
            attention_2 = self.softmax(dot_2.squeeze(2)).unsqueeze(2)
            enc_attention_2 = torch.bmm(enc_2.permute(0, 2, 1), attention_2)

        if self.separate_attention == "separate_different_encoder_type":
            hid = F.tanh(self.linear_att(torch.cat(
                (enc_attention.squeeze(2), enc_attention_2.squeeze(2), dec_s_top), 1)))
        else:
            hid = F.tanh(self.linear_att(
                torch.cat((enc_attention.squeeze(2), dec_s_top), 1)))
        h2y_in = hid
        if self.opt.dropout_for_predict > 0:
            h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)

        return pred

class Dec_LSTM(nn.Module):
    '''
    Decoder LSTM cell with parent hidden state
    '''

    def __init__(self, input_size, hidden_size, dropout):
        super(Dec_LSTM, self).__init__()
        self.rnn_size = hidden_size
        self.word_embedding_size = input_size
        self.i2h = nn.Linear(self.word_embedding_size+self.rnn_size, 4*self.rnn_size)

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
        if self.opt.dropout_de_out > 0:
            cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return cy, hy


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_embedding_size = input_size

        self.lstm = Dec_LSTM(input_size, hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_src, prev_c, prev_h, parent_h):

        src_emb = self.embedding(input_src)
        src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(
            src_emb, prev_c, prev_h, parent_h)
        return prev_cy, prev_hy

def get_dec_batch(dec_tree_batch, batch_size, using_gpu, form_manager):
    queue_tree = {}
    for i in range(1, batch_size+1):
        queue_tree[i] = []
        queue_tree[i].append({"tree" : dec_tree_batch[i-1], "parent": 0, "child_index": 1})

    cur_index, max_index = 1,1
    dec_batch = {}
    # max_index: the max number of sequence decoder in one batch
    while (cur_index <= max_index):
        max_w_len = -1
        batch_w_list = []
        for i in range(1, batch_size+1):
            w_list = []
            if (cur_index <= len(queue_tree[i])):
                t = queue_tree[i][cur_index - 1]["tree"]

                for ic in range (t.num_children):
                    if isinstance(t.children[ic], Tree):
                        w_list.append(4)
                        queue_tree[i].append({"tree" : t.children[ic], "parent" : cur_index, "child_index": ic + 1})
                    else:
                        w_list.append(t.children[ic])
                if len(queue_tree[i]) > max_index:
                    max_index = len(queue_tree[i])
            if len(w_list) > max_w_len:
                max_w_len = len(w_list)
            batch_w_list.append(w_list)
        dec_batch[cur_index] = torch.zeros((batch_size, max_w_len + 2), dtype=torch.long)
        for i in range(batch_size):
            w_list = batch_w_list[i]
            if len(w_list) > 0:
                for j in range(len(w_list)):
                    dec_batch[cur_index][i][j+1] = w_list[j]
                # add <S>, <E>
                if cur_index == 1:
                    dec_batch[cur_index][i][0] = 1
                else:
                    dec_batch[cur_index][i][0] = form_manager.get_symbol_idx('(')
                dec_batch[cur_index][i][len(w_list) + 1] = 2

        if using_gpu:
            dec_batch[cur_index] = dec_batch[cur_index].cuda()
        cur_index += 1

    return dec_batch, queue_tree, max_index
