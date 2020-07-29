import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dec_LSTM(nn.Module):
    '''
    Decoder LSTM cell with parent hidden state
    '''

    def __init__(self, opt):
        super(Dec_LSTM, self).__init__()
        self.opt = opt
        self.word_embedding_size = 300
        self.i2h = nn.Linear(self.word_embedding_size+2 *
                             opt.rnn_size, 4*opt.rnn_size)
        self.h2h = nn.Linear(opt.rnn_size, 4*opt.rnn_size)

        if opt.dropout_de_out > 0:
            self.dropout = nn.Dropout(opt.dropout_de_out)

    def forward(self, x, prev_c, prev_h, parent_h, sibling_state):
        input_cat = torch.cat((x, parent_h, sibling_state), 1)
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
    def __init__(self, opt, input_size):
        super(DecoderRNN, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size
        self.word_embedding_size = 300
        self.embedding = nn.Embedding(
            input_size, self.word_embedding_size, padding_idx=0)

        self.lstm = Dec_LSTM(self.opt)
        if opt.dropout_de_in > 0:
            self.dropout = nn.Dropout(opt.dropout_de_in)

    def forward(self, input_src, prev_c, prev_h, parent_h, sibling_state):

        src_emb = self.embedding(input_src)
        if self.opt.dropout_de_in > 0:
            src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(
            src_emb, prev_c, prev_h, parent_h, sibling_state)
        return prev_cy, prev_hy


class AttnUnit(nn.Module):
    def __init__(self, opt, output_size):
        super(AttnUnit, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size
        self.separate_attention = True
        if self.separate_attention:
            self.linear_att = nn.Linear(3*self.hidden_size, self.hidden_size)
        else:
            self.linear_att = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.linear_out = nn.Linear(self.hidden_size, output_size)
        if opt.dropout_for_predict > 0:
            self.dropout = nn.Dropout(opt.dropout_for_predict)

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top, enc_2):
        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        enc_attention = torch.bmm(enc_s_top.permute(0, 2, 1), attention)

        if self.separate_attention:
            dot_2 = torch.bmm(enc_2, dec_s_top.unsqueeze(2))
            attention_2 = self.softmax(dot_2.squeeze(2)).unsqueeze(2)
            enc_attention_2 = torch.bmm(enc_2.permute(0, 2, 1), attention_2)

        if self.separate_attention:
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
