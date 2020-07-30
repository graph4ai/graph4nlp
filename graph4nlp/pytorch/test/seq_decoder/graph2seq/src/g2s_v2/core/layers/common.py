'''
Created on Nov, 2018

@author: hugo

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from graph4nlp.pytorch.test.seq_decoder.graph2seq.src.g2s_v2.core.layers.attention import Attention
from graph4nlp.pytorch.test.seq_decoder.graph2seq.src.g2s_v2.core.utils.constants import VERY_SMALL_NUMBER
from graph4nlp.pytorch.test.seq_decoder.graph2seq.src.g2s_v2.core.utils.generic_utils import to_cuda


class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        '''GatedFusion module'''
        self.fc_z = nn.Linear(4 * hidden_size, hidden_size, bias=True)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.fc_z(torch.cat([h_state, input, h_state * input, h_state - input], -1)))
        h_state = (1 - z) * h_state + z * input
        return h_state


class GRUStep(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(GRUStep, self).__init__()
        '''GRU module'''
        self.linear_z = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_r = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_t = nn.Linear(hidden_size + input_size, hidden_size, bias=False)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input], -1)))
        r = torch.sigmoid(self.linear_r(torch.cat([h_state, input], -1)))
        t = torch.tanh(self.linear_t(torch.cat([r * h_state, input], -1)))
        h_state = (1 - z) * h_state + z * t
        return h_state


def dropout(x, drop_prob, shared_axes=[], training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    input_tensor: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
    """
    if drop_prob == 0 or drop_prob == None or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, \
                 bidirectional=False, num_layers=1, rnn_type='lstm', rnn_dropout=None, device=None):
        super(EncoderRNN, self).__init__()
        if not rnn_type in ('lstm', 'gru'):
            raise RuntimeError('rnn_type is expected to be lstm or gru, got {}'.format(rnn_type))
        if bidirectional:
            print('[ Using {}-layer bidirectional {} encoder ]'.format(num_layers, rnn_type))
        else:
            print('[ Using {}-layer {} encoder ]'.format(num_layers, rnn_type))
        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout
        self.device = device
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """x: [batch_size * max_length * emb_dim]
           x_len: [batch_size]
        """
        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)

        h0 = to_cuda(torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size), self.device)
        if self.rnn_type == 'lstm':
            c0 = to_cuda(torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size),
                         self.device)
            packed_h, (packed_h_t, packed_c_t) = self.model(x, (h0, c0))
        else:
            packed_h, packed_h_t = self.model(x, h0)

        if self.num_directions == 2:
            packed_h_t = torch.cat((packed_h_t[-1], packed_h_t[-2]), 1)
            if self.rnn_type == 'lstm':
                packed_c_t = torch.cat((packed_c_t[-1], packed_c_t[-2]), 1)
        else:
            packed_h_t = packed_h_t[-1]
            if self.rnn_type == 'lstm':
                packed_c_t = packed_c_t[-1]

        # restore the sorting
        _, inverse_indx = torch.sort(indx, 0)

        hh, _ = pad_packed_sequence(packed_h, batch_first=True)
        restore_hh = hh[inverse_indx]
        restore_hh = dropout(restore_hh, self.rnn_dropout, shared_axes=[-2], training=self.training)
        restore_hh = restore_hh.transpose(0, 1)  # [max_length, batch_size, emb_dim]

        restore_packed_h_t = packed_h_t[inverse_indx]
        restore_packed_h_t = dropout(restore_packed_h_t, self.rnn_dropout, training=self.training)
        restore_packed_h_t = restore_packed_h_t.unsqueeze(0)  # [1, batch_size, emb_dim]

        if self.rnn_type == 'lstm':
            restore_packed_c_t = packed_c_t[inverse_indx]
            restore_packed_c_t = dropout(restore_packed_c_t, self.rnn_dropout, training=self.training)
            restore_packed_c_t = restore_packed_c_t.unsqueeze(0)  # [1, batch_size, emb_dim]
            rnn_state_t = (restore_packed_h_t, restore_packed_c_t)
        else:
            rnn_state_t = restore_packed_h_t
        return restore_hh, rnn_state_t


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, *, rnn_type='lstm', enc_attn=True, dec_attn=True,
                 enc_attn_cover=True, separate_attn=False, pointer=True, tied_embedding=None, out_embed_size=None,
                 in_drop: float = 0, rnn_drop: float = 0, out_drop: float = 0, enc_hidden_size=None, device=None):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.in_drop = in_drop
        self.out_drop = out_drop
        self.rnn_drop = rnn_drop
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.combined_size = self.hidden_size
        self.enc_attn = enc_attn
        self.dec_attn = dec_attn
        self.enc_attn_cover = enc_attn_cover
        self.pointer = pointer
        self.separate_attn = separate_attn

        self.out_embed_size = out_embed_size
        if tied_embedding is not None and self.out_embed_size and embed_size != self.out_embed_size:
            print("Warning: Output embedding size %d is overriden by its tied embedding size %d."
                  % (self.out_embed_size, embed_size))
            self.out_embed_size = embed_size

        model = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.model = model(embed_size, self.hidden_size)

        if enc_attn:
            if self.separate_attn:
                num_attn = 2
                self.enc_attn_fn2 = Attention(self.hidden_size, 2 * self.hidden_size, self.hidden_size, attn_type='add')

            else:
                num_attn = 1

            if not enc_hidden_size: enc_hidden_size = self.hidden_size
            self.fc_dec_input = nn.Linear(num_attn * enc_hidden_size + embed_size, embed_size)
            self.enc_attn_fn = Attention(self.hidden_size, 2 * self.hidden_size, self.hidden_size, attn_type='add')
            self.combined_size += num_attn * enc_hidden_size
            if enc_attn_cover:
                self.cover_weight = torch.Tensor(1, 1, self.hidden_size)
                self.cover_weight = nn.Parameter(nn.init.xavier_uniform_(self.cover_weight))

        if dec_attn:
            self.dec_attn_fn = Attention(self.hidden_size, 2 * self.hidden_size, self.hidden_size, attn_type='add')
            self.combined_size += self.hidden_size

        if pointer:
            self.ptr = nn.Linear(self.combined_size + embed_size + self.hidden_size, 1)

        if tied_embedding is not None and embed_size != self.combined_size:
            # use pre_out layer if combined size is different from embedding size
            self.out_embed_size = embed_size

        if self.out_embed_size:  # use pre_out layer
            self.pre_out = nn.Linear(self.combined_size, self.out_embed_size, bias=False)
            size_before_output = self.out_embed_size
        else:  # don't use pre_out layer
            size_before_output = self.combined_size

        self.out = nn.Linear(size_before_output, vocab_size, bias=False)
        if tied_embedding is not None:
            self.out.weight = tied_embedding.weight

    def forward(self, embedded, rnn_state, encoder_hiddens=None, decoder_hiddens=None, coverage_vector=None, *,
                input_mask=None, encoder_word_idx=None, ext_vocab_size: int = None, log_prob: bool = True,
                prev_enc_context=None, encoder_outputs2=None):
        """
        :param embedded: (batch size, embed size)
        :param rnn_state: LSTM: ((1, batch size, decoder hidden size), (1, batch size, decoder hidden size)), GRU:(1, batch size, decoder hidden size)
        :param encoder_hiddens: (src seq len, batch size, hidden size), for attention mechanism
        :param decoder_hiddens: (past dec steps, batch size, hidden size), for attention mechanism
        :param encoder_word_idx: (batch size, src seq len), for pointer network
        :param ext_vocab_size: the dynamic word_vocab size, determined by the max num of OOV words contained
                               in any src seq in this batch, for pointer network
        :param log_prob: return log probability instead of probability
        :return: tuple of four things:
                 1. word prob or log word prob, (batch size, dynamic word_vocab size);
                 2. rnn_state, RNN hidden (and/or ceil) state after this step, (1, batch size, decoder hidden size);
                 3. attention weights over encoder states, (batch size, src seq len);
                 4. prob of copying by pointing as opposed to generating, (batch size, 1)

        Perform single-step decoding.
        """
        batch_size = embedded.size(0)
        combined = to_cuda(torch.zeros(batch_size, self.combined_size), self.device)

        embedded = dropout(embedded, self.in_drop, training=self.training)

        if self.enc_attn:
            if prev_enc_context is None:
                num_attn = 2 if self.separate_attn else 1
                prev_enc_context = to_cuda(torch.zeros(batch_size, num_attn * encoder_hiddens.size(-1)), self.device)

            dec_input_emb = self.fc_dec_input(torch.cat([embedded, prev_enc_context], -1))
        else:
            dec_input_emb = embedded

        output, rnn_state = self.model(dec_input_emb.unsqueeze(0), rnn_state)  # unsqueeze and squeeze are necessary
        output = dropout(output, self.rnn_drop, training=self.training)
        if self.rnn_type == 'lstm':
            rnn_state = tuple([dropout(x, self.rnn_drop, training=self.training) for x in rnn_state])
            hidden = torch.cat(rnn_state, -1).squeeze(0)
        else:
            rnn_state = dropout(rnn_state, self.rnn_drop, training=self.training)
            hidden = rnn_state.squeeze(0)

        combined[:, :self.hidden_size] = output.squeeze(0)  # as RNN expects a 3D tensor (step=1)
        offset = self.hidden_size
        enc_attn, prob_ptr = None, None  # for visualization

        if self.enc_attn or self.pointer:
            # energy and attention: (num encoder states, batch size, 1)
            num_enc_steps = encoder_hiddens.size(0)
            enc_total_size = encoder_hiddens.size(2)
            if self.separate_attn:
                enc_total_size += encoder_outputs2.size(2)

            if self.enc_attn_cover and coverage_vector is not None:
                # Shape (batch size, num encoder states, encoder hidden size)
                addition_vec = coverage_vector.unsqueeze(-1) * self.cover_weight
            else:
                addition_vec = None
            enc_energy = self.enc_attn_fn(hidden, encoder_hiddens.transpose(0, 1).contiguous(), \
                                          attn_mask=input_mask, addition_vec=addition_vec).transpose(0, 1).unsqueeze(-1)

            # transpose => (batch size, num encoder states, 1)
            enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)
            if self.enc_attn:
                # context: (batch size, encoder hidden size, 1)
                enc_context = torch.bmm(encoder_hiddens.permute(1, 2, 0), enc_attn).squeeze(2)

                if self.separate_attn:
                    enc_energy2 = self.enc_attn_fn2(hidden, encoder_outputs2.transpose(0, 1).contiguous(), \
                                                    attn_mask=input_mask, addition_vec=addition_vec).transpose(0,
                                                                                                               1).unsqueeze(
                        -1)

                    # transpose => (batch size, num encoder states, 1)
                    enc_attn2 = F.softmax(enc_energy2, dim=0).transpose(0, 1)

                    # context: (batch size, encoder hidden size, 1)
                    enc_context2 = torch.bmm(encoder_outputs2.permute(1, 2, 0), enc_attn2).squeeze(2)
                    enc_context = torch.cat([enc_context, enc_context2], 1)

                combined[:, offset:offset + enc_total_size] = enc_context
                offset += enc_total_size
            else:
                enc_context = None

            if self.separate_attn:
                enc_attn = (enc_attn.squeeze(2) + enc_attn2.squeeze(2)) / 2
            else:
                enc_attn = enc_attn.squeeze(2)

        if self.dec_attn:
            if decoder_hiddens is not None and len(decoder_hiddens) > 0:
                dec_energy = self.dec_attn_fn(hidden, decoder_hiddens \
                                              .transpose(0, 1).contiguous()).transpose(0, 1).unsqueeze(-1)

                dec_attn = F.softmax(dec_energy, dim=0).transpose(0, 1)
                dec_context = torch.bmm(decoder_hiddens.permute(1, 2, 0), dec_attn)
                combined[:, offset:offset + self.hidden_size] = dec_context.squeeze(2)
            offset += self.hidden_size

        # generator
        if self.out_embed_size:
            out_embed = torch.tanh(self.pre_out(combined))
        else:
            out_embed = combined
        out_embed = dropout(out_embed, self.out_drop, training=self.training)

        logits = self.out(out_embed)  # (batch size, word_vocab size)

        # pointer
        if self.pointer:
            output = to_cuda(torch.zeros(batch_size, ext_vocab_size), self.device)
            # distribute probabilities between generator and pointer
            pgen_cat = [embedded, hidden]
            if self.enc_attn:
                pgen_cat.append(enc_context)
            if self.dec_attn:
                pgen_cat.append(dec_context)

            prob_ptr = torch.sigmoid(self.ptr(torch.cat(pgen_cat, -1)))  # (batch size, 1)
            prob_gen = 1 - prob_ptr
            # add generator probabilities to output
            gen_output = F.softmax(logits, dim=1)  # can't use log_softmax due to adding probabilities

            output[:, :self.vocab_size] = prob_gen * gen_output
            # add pointer probabilities to output
            ptr_output = enc_attn
            output.scatter_add_(1, encoder_word_idx, prob_ptr * ptr_output)
            if log_prob: output = torch.log(output + VERY_SMALL_NUMBER)
        else:
            if log_prob:
                output = F.log_softmax(logits, dim=1)
            else:
                output = F.softmax(logits, dim=1)

        return output, rnn_state, enc_attn, prob_ptr, enc_context
