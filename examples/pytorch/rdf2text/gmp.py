import torch
import torch.nn as nn
import numpy as np

def init_wt_normal(wt, config):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_lstm_wt(lstm, config):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


class GMPEncoder(nn.Module):
    def __init__(self, 
                 input_size,
                 hidden_size,
                 output_size,
                 word_emb,
                 bigtr=True):
        super(GMPEncoder, self).__init__()
        # self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        # init_wt_normal(self.embedding.weight)
        
        self.bigtr = bigtr
        self.hidden_size = hidden_size
        self.embedding = word_emb

        if bigtr == True:
            self.lstm_s2t = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                                    bidirectional=False)
            self.lstm_t2s = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                                    bidirectional=False)
            self.W_h = nn.Linear(hidden_size*2, hidden_size*2, bias=False)
        else:
            self.lstm_s2t = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
            self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # init_lstm_wt(self.lstm_s2t)

    def forward_onestep(self, x_t_1, s_t_1):
        embedded = self.embedding(x_t_1)
        lstm_out, s_t = self.lstm_s2t(embedded.unsqueeze(1), s_t_1)

        return lstm_out, s_t

    def forward_onestep_fw(self, x_t_1, s_t_1):
        embedded = self.embedding(x_t_1)
        lstm_out, s_t = self.lstm_s2t(embedded.unsqueeze(1), s_t_1)

        return lstm_out, s_t

    def forward_onestep_bw(self, x_t_1, s_t_1):
        embedded = self.embedding(x_t_1)
        lstm_out, s_t = self.lstm_t2s(embedded.unsqueeze(1), s_t_1)

        return lstm_out, s_t

    # def forward(self, src_seqs, src_lengths, src_jump, src_seqs_rev=None, src_jump_rev=None):
    def forward(self, src_seqs, src_jump):
        """

        :param input: [batch_size, max_seq_len]
        :return:
        """

        if self.bigtr == True:
            # if config.use_cuda==True:
            #     src_seqs_rev = torch.tensor(np.flip(src_seqs.detach().cpu().numpy(),1).copy()).cuda()
            #     src_jump_rev = torch.tensor(np.flip(src_jump.detach().cpu().numpy(),1).copy()).cuda()
            #     h_t_0 = torch.zeros((1, src_seqs.size()[0], self.hidden_size)).cuda()  # (2, b, h)
            #     c_t_0 = torch.zeros((1, src_seqs.size()[0], self.hidden_size)).cuda()
            # else:
            src_seqs_rev = torch.Tensor(np.flip(src_seqs.detach().cpu().numpy(), 1).copy()).long().to(src_seqs.device)
            src_jump_rev = torch.Tensor(np.flip(src_jump.detach().cpu().numpy(), 1).copy()).long().to(src_seqs.device)
            h_t_0 = torch.zeros((1, src_seqs.size()[0], self.hidden_size)).to(src_seqs.device) # (2, b, h)
            c_t_0 = torch.zeros((1, src_seqs.size()[0], self.hidden_size)).to(src_seqs.device)


            h_memory_fw = []
            h_memory_bw = []
            c_memory_fw = []
            c_memory_bw = []
            h_t_1 = h_t_0
            c_t_1 = c_t_0
            outputs_fw = []
            outputs_bw = []

            for di in range(src_seqs.size()[1]):
                x_t_1 = src_seqs[:, di]
                s_t_1 = (h_t_1, c_t_1)
                out_t_1, (h_t_1, c_t_1) = self.forward_onestep_fw(x_t_1, s_t_1)
                h_memory_fw.append(h_t_1)
                c_memory_fw.append(c_t_1)
                h_t_1 = h_t_1.squeeze() * src_jump[:,di].unsqueeze(1)
                c_t_1 = c_t_1.squeeze() * src_jump[:,di].unsqueeze(1)
                h_t_1 = h_t_1.unsqueeze(0)
                c_t_1 = c_t_1.unsqueeze(0)

                outputs_fw.append(out_t_1)

            h_t_1 = h_t_0
            c_t_1 = c_t_0
            for di in range(src_seqs_rev.size()[1]):
                x_t_1 = src_seqs_rev[:, di]
                s_t_1 = (h_t_1, c_t_1)
                out_t_1, (h_t_1, c_t_1) = self.forward_onestep_bw(x_t_1, s_t_1)
                h_memory_bw.append(h_t_1)
                c_memory_bw.append(c_t_1)
                h_t_1 = h_t_1.squeeze() * src_jump_rev[:, di].unsqueeze(1)
                c_t_1 = c_t_1.squeeze() * src_jump_rev[:, di].unsqueeze(1)
                h_t_1 = h_t_1.unsqueeze(0)
                c_t_1 = c_t_1.unsqueeze(0)

                outputs_bw.append(out_t_1)

            enc_padding_mask = torch.gt(src_seqs, 0).float()  # (batch_size, seq_len)
            enc_padding_mask_rev = torch.gt(src_seqs_rev, 0).float()  # (batch_size, seq_len)
            encoder_outputs_fw = torch.cat(outputs_fw, dim=1)
            encoder_outputs_bw = torch.cat(outputs_bw, dim=1)
            encoder_outputs_fw = encoder_outputs_fw * enc_padding_mask.unsqueeze(2)
            encoder_outputs_bw = encoder_outputs_bw * enc_padding_mask_rev.unsqueeze(2)
            # encoder_outputs_fw = np.flip(encoder_outputs_fw.detach().cpu().numpy(),1).copy()
            encoder_outputs_bw = torch.Tensor(np.flip(encoder_outputs_bw.detach().cpu().numpy(),1).copy()).to(src_seqs.device)
            outputs = torch.cat((encoder_outputs_fw,encoder_outputs_bw),dim=2)

            h_memory_tensor_fw = torch.cat(h_memory_fw, dim=0).permute(1, 0, 2)
            h_memory_tensor_bw = torch.cat(h_memory_bw, dim=0).permute(1, 0, 2)
            c_memory_tensor_fw = torch.cat(c_memory_fw, dim=0).permute(1, 0, 2)
            c_memory_tensor_bw = torch.cat(c_memory_bw, dim=0).permute(1, 0, 2)
            # h_memory_tensor_bw = h_memory_tensor_bw * enc_padding_mask.unsqueeze(2)
            # c_memory_tensor = c_memory_tensor * enc_padding_mask.unsqueeze(2)

            h_max_fw = torch.max(h_memory_tensor_fw, dim=1)[0].unsqueeze(0)
            h_max_bw = torch.max(h_memory_tensor_bw, dim=1)[0].unsqueeze(0)
            c_max_fw = torch.max(c_memory_tensor_fw, dim=1)[0].unsqueeze(0)
            c_max_bw = torch.max(c_memory_tensor_bw, dim=1)[0].unsqueeze(0)

            h_max = torch.cat((h_max_fw, h_max_bw), dim=0)
            c_max = torch.cat((c_max_fw, c_max_bw), dim=0)

            encoder_feature = outputs.view(-1, 2 * self.hidden_size)  # B * t_k x 2*hidden_dim
            encoder_feature = self.W_h(encoder_feature)

            enc_padding_mask = enc_padding_mask.masked_fill(enc_padding_mask == 0., -1.)

            return outputs, encoder_feature, (h_max, c_max), enc_padding_mask
        else:
            h_t_0 = torch.zeros((1, src_seqs.size()[0], self.hidden_size)).cuda()  # (1, b, h)
            c_t_0 = torch.zeros((1, src_seqs.size()[0], self.hidden_size)).cuda()
            h_memory = []
            c_memory = []
            h_t_1 = h_t_0
            c_t_1 = c_t_0
            outputs = []
            for di in range(src_seqs.size()[1]):
                x_t_1 = src_seqs[:,di]
                s_t_1 = (h_t_1, c_t_1)
                out_t_1, (h_t_1, c_t_1) = self.forward_onestep(x_t_1, s_t_1)
                h_memory.append(h_t_1)
                c_memory.append(c_t_1)
                if self.bigtr==True:
                    h_t_1 = h_t_1 * src_jump[:,di].unsqueeze(1)
                    c_t_1 = c_t_1 * src_jump[:,di].unsqueeze(1)
                else:
                    h_t_1 = h_t_1.squeeze() * src_jump[:, di].unsqueeze(1)
                    c_t_1 = c_t_1.squeeze() * src_jump[:, di].unsqueeze(1)
                    h_t_1 = h_t_1.unsqueeze(0)
                    c_t_1 = c_t_1.unsqueeze(0)
                outputs.append(out_t_1)

            enc_padding_mask = torch.gt(src_seqs, 0).float()  # (batch_size, seq_len)
            encoder_outputs = torch.cat(outputs,dim=1)
            encoder_outputs = encoder_outputs*enc_padding_mask.unsqueeze(2)

            h_memory_tensor = torch.cat(h_memory, dim=0).permute(1, 0, 2)
            c_memory_tensor = torch.cat(c_memory, dim=0).permute(1, 0, 2)
            h_memory_tensor = h_memory_tensor * enc_padding_mask.unsqueeze(2)
            c_memory_tensor = c_memory_tensor * enc_padding_mask.unsqueeze(2)

            h_max = torch.max(h_memory_tensor, dim=1)[0].unsqueeze(0)
            c_max = torch.max(c_memory_tensor, dim=1)[0].unsqueeze(0)

            encoder_feature = encoder_outputs.view(-1, self.hidden_size)  # B * t_k x 2*hidden_dim
            encoder_feature = self.W_h(encoder_feature)


        if self.bigtr == True:
            h_max = torch.stack(h_memory).max(dim=0)[0]
            c_max = torch.stack(c_memory).max(dim=0)[0]
            encoder_feature = encoder_outputs.view(-1, 2*self.hidden_size)  # B * t_k x 2*hidden_dim

        return encoder_outputs, encoder_feature, (h_max, c_max)