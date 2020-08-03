import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from .base import RNNTreeDecoderBase
from ...utils.tree_utils import Tree, to_cuda


class StdTreeDecoder(RNNTreeDecoderBase):
    def __init__(self, attn, attn_type, embeddings, enc_hidden_size, dec_emb_size, dec_hidden_size, output_size, device, criterion, teacher_force_ratio, use_sibling = True, use_attention=True, use_copy=False, use_coverage=False,
                 fuse_strategy="average", num_layers=1, dropout_input=0.1, dropout_output=0.3, rnn_type="lstm", max_dec_seq_length=512, max_dec_tree_depth=256, tgt_vocab=None):
        super(StdTreeDecoder, self).__init__(use_attention=True, use_copy=use_copy, use_coverage=False, attention_type="uniform",
                                             fuse_strategy="average")
        self.num_layers = num_layers
        self.device = device
        self.criterion = criterion
        self.rnn_size = dec_hidden_size
        self.enc_hidden_size = enc_hidden_size
        self.hidden_size = dec_hidden_size
        self.max_dec_seq_length = max_dec_seq_length
        self.max_dec_tree_depth = max_dec_tree_depth
        self.tgt_vocab = tgt_vocab
        self.teacher_force_ratio = teacher_force_ratio
        self.use_sibling = use_sibling
        self.dec_emb_size = dec_emb_size
        self.dropout_input = dropout_input
        self.dropout_output = dropout_output
        self.embeddings = embeddings
                
        self.attn_state = {}
        self.use_coverage = use_coverage
        self.use_copy = use_copy
        self.attention = attn
        self.separate_attn = (attn_type != "uniform")

        self.rnn = self._build_rnn(rnn_type=rnn_type, input_size=output_size, emb_size=dec_emb_size, hidden_size=dec_hidden_size, dropout_input=dropout_input, dropout_output=dropout_output, use_sibling = use_sibling, device=device)


    def _run_forward_pass(self, graph_node_embedding, graph_node_mask, rnn_node_embedding, graph_level_embedding,
                          graph_edge_embedding=None, graph_edge_mask=None, tgt_tree_batch=None, enc_batch=None):
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

        enc_outputs = rnn_node_embedding
        graph_hidden_state, graph_cell_state = graph_level_embedding
        # graph_cell_state = graph_level_embedding
        # graph_hidden_state = graph_level_embedding

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
                    (tgt_batch_size, self.rnn_size), dtype=torch.float, requires_grad=False, device=self.device)
                to_cuda(dec_state[cur_index][0][j], self.device)

            sibling_state = torch.zeros(
                (tgt_batch_size, self.rnn_size), dtype=torch.float, requires_grad=False)
            to_cuda(sibling_state, self.device)
            
            # with torch.no_grad():
            if cur_index == 1:
                for i in range(tgt_batch_size):
                    dec_state[1][0][1][i, :] = graph_cell_state[i]
                    dec_state[1][0][2][i, :] = graph_hidden_state[i]

            else:
                for i in range(1, tgt_batch_size+1):
                    if (cur_index <= len(queue_tree[i])):
                        par_index = queue_tree[i][cur_index - 1]["parent"]
                        child_index = queue_tree[i][cur_index -
                                                    1]["child_index"]

                        dec_state[cur_index][0][1][i-1, :] = dec_state[par_index][child_index][1][i-1, :]
                        dec_state[cur_index][0][2][i-1, :] = dec_state[par_index][child_index][2][i-1, :]

                flag_sibling = False
                for q_index in range(len(queue_tree[i])):
                    if (cur_index <= len(queue_tree[i])) and (q_index < cur_index - 1) and (queue_tree[i][q_index]["parent"] == queue_tree[i][cur_index - 1]["parent"]) and (queue_tree[i][q_index]["child_index"] < queue_tree[i][cur_index - 1]["child_index"]):
                        flag_sibling = True
                        sibling_index = q_index
                if flag_sibling:
                    sibling_state[i - 1, :] = dec_state[sibling_index][dec_batch[sibling_index].size(1) - 1][2][i - 1, :]

            if self.use_copy:
                enc_context = None
                input_mask = create_mask(torch.LongTensor([enc_outputs.size(1)]*tgt_batch_size), enc_outputs.size(1), self.device)
                decoder_state = (dec_state[cur_index][0][1].unsqueeze(0), dec_state[cur_index][0][2].unsqueeze(0))
                parent_h = dec_state[cur_index][0][2]

                for i in range(dec_batch[cur_index].size(1) - 1):
                    teacher_force = random.random() < self.teacher_force_ratio
                    if teacher_force != True and i > 0:
                        input_word = pred.argmax(1)
                    else:
                        input_word = dec_batch[cur_index][:, i]
                    decoder_embedded = self.embeddings(input_word)
                    pred, decoder_state, _, _, enc_context = self.rnn(parent_h, sibling_state, decoder_embedded,
                                                  decoder_state,
                                                  enc_outputs.transpose(0,1),
                                                  None, None, input_mask=input_mask,
                                                  encoder_word_idx=enc_batch,
                                                  ext_vocab_size=self.embeddings.num_embeddings,
                                                  log_prob=False,
                                                  prev_enc_context=enc_context,
                                                  encoder_outputs2=rnn_node_embedding.transpose(0,1))
                    dec_next_state_1, dec_next_state_2 = decoder_state
                    dec_state[cur_index][i+1][1] = dec_next_state_1.squeeze(0)
                    dec_state[cur_index][i+1][2] = dec_next_state_2.squeeze(0)

                    pred = torch.log(pred + 1e-31)
                    loss += self.criterion(pred, dec_batch[cur_index][:, i+1])
            else:
                parent_h = dec_state[cur_index][0][2]
                for i in range(dec_batch[cur_index].size(1) - 1):
                    teacher_force = random.random() < self.teacher_force_ratio
                    if teacher_force != True and i > 0:
                        input_word = pred.argmax(1)
                    else:
                        input_word = dec_batch[cur_index][:, i]

                    dec_state[cur_index][i+1][1], dec_state[cur_index][i+1][2] = self.rnn(
                        input_word, dec_state[cur_index][i][1], dec_state[cur_index][i][2], parent_h, sibling_state)
                    # print(enc_outputs.is_leaf)
                    pred = self.attention(
                        enc_outputs, dec_state[cur_index][i+1][2], torch.tensor(0))
                    # output_tree_batch
                    # print(pred.is_leaf)
                    # print(i)
                    loss += self.criterion(pred, dec_batch[cur_index][:, i+1])
            cur_index = cur_index + 1
        loss = loss / tgt_batch_size
        return loss

    def _build_rnn(self, rnn_type, input_size, emb_size, hidden_size, dropout_input, dropout_output, use_sibling, device):
        if not self.use_copy:
            rnn = DecoderRNN(input_size, emb_size, hidden_size, dropout_input, dropout_output, use_sibling)
        else:
            rnn = DecoderRNNWithCopy(input_size, emb_size, hidden_size, rnn_type=rnn_type,
                             enc_attn=True, dec_attn=False, separate_attn=self.separate_attn,
                             pointer=self.use_copy, out_embed_size=None,
                             tied_embedding=self.embeddings,
                             in_drop=self.dropout_input, rnn_drop=self.dropout_output,
                             out_drop=0, enc_hidden_size=None, device=device, use_sibling=use_sibling)
        return rnn


class DecoderRNNWithCopy(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, *, rnn_type='lstm', enc_attn=True, dec_attn=True,
               enc_attn_cover=True, separate_attn=False, pointer=True, tied_embedding=None, out_embed_size=None,
               in_drop: float=0, rnn_drop: float=0, out_drop: float=0, enc_hidden_size=None, device=None, use_sibling=False):
        super(DecoderRNNWithCopy, self).__init__()
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
        self.word_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.use_sibling = use_sibling

        self.out_embed_size = out_embed_size
        if tied_embedding is not None and self.out_embed_size and embed_size != self.out_embed_size:
            print("Warning: Output embedding size %d is overriden by its tied embedding size %d."
                % (self.out_embed_size, embed_size))
            self.out_embed_size = embed_size

        model = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU

        if self.use_sibling:
            self.model = model(embed_size + hidden_size*2, self.hidden_size)
        else:
            self.model = model(embed_size + hidden_size, self.hidden_size)
        # print(embed_size, self.hidden_size)
        # print(separate_attn)
        if enc_attn:
            if self.separate_attn:
                num_attn = 2
                self.enc_attn_fn2 = Attention(self.hidden_size, 2 * self.hidden_size, self.hidden_size, attn_type='add')

            else:
                num_attn = 1

            if not enc_hidden_size: enc_hidden_size = self.hidden_size

            if self.use_sibling:
                self.fc_dec_input = nn.Linear(num_attn * enc_hidden_size + embed_size + hidden_size*2, embed_size + hidden_size*2)
            else:
                self.fc_dec_input = nn.Linear(num_attn * enc_hidden_size + embed_size + hidden_size, embed_size + hidden_size)

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

    def forward(self, parent_h, sibling_state, embedded, rnn_state, encoder_hiddens=None, decoder_hiddens=None, coverage_vector=None, *,
              input_mask=None, encoder_word_idx=None, ext_vocab_size: int=None, log_prob: bool=True, prev_enc_context=None, encoder_outputs2=None):
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
        
        combined = torch.zeros(batch_size, self.combined_size).to(self.device)

        embedded = dropout(embedded, self.in_drop, training=self.training)

        if self.enc_attn:
            if prev_enc_context is None:
                num_attn = 2 if self.separate_attn else 1
                prev_enc_context = torch.zeros(batch_size, num_attn * encoder_hiddens.size(-1)).to(self.device)
                # print(num_attn, embedded.size(), prev_enc_context.size())
            if self.use_sibling:
                dec_input_emb = self.fc_dec_input(torch.cat([embedded, parent_h, sibling_state, prev_enc_context], -1))
            else:
                dec_input_emb = self.fc_dec_input(torch.cat([embedded, parent_h, prev_enc_context], -1))
        else:
            if self.use_sibling:
                dec_input_emb = torch.cat([embedded, parent_h, sibling_state], -1)
            else:
                dec_input_emb = torch.cat([embedded, parent_h], -1)

        output, rnn_state = self.model(dec_input_emb.unsqueeze(0), rnn_state) # unsqueeze and squeeze are necessary
        output = dropout(output, self.rnn_drop, training=self.training)
        if self.rnn_type == 'lstm':
            rnn_state = tuple([dropout(x, self.rnn_drop, training=self.training) for x in rnn_state])
            hidden = torch.cat(rnn_state, -1).squeeze(0)
        else:
            rnn_state = dropout(rnn_state, self.rnn_drop, training=self.training)
            hidden = rnn_state.squeeze(0)

        combined[:, :self.hidden_size] = output.squeeze(0)        # as RNN expects a 3D tensor (step=1)
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
                                attn_mask=input_mask, addition_vec=addition_vec).transpose(0, 1).unsqueeze(-1)

                    # transpose => (batch size, num encoder states, 1)
                    enc_attn2 = F.softmax(enc_energy2, dim=0).transpose(0, 1)

                    # context: (batch size, encoder hidden size, 1)
                    enc_context2 = torch.bmm(encoder_outputs2.permute(1, 2, 0), enc_attn2).squeeze(2)
                    enc_context = torch.cat([enc_context, enc_context2], 1)

                combined[:, offset:offset+enc_total_size] = enc_context
                offset += enc_total_size
            else:
                enc_context = None

            if self.separate_attn:
                enc_attn = (enc_attn.squeeze(2) + enc_attn2.squeeze(2)) / 2
            else:
                enc_attn = enc_attn.squeeze(2)


        if self.dec_attn:
            if decoder_hiddens is not None and len(decoder_hiddens) > 0:
                dec_energy = self.dec_attn_fn(hidden, decoder_hiddens\
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
            output = torch.zeros(batch_size, ext_vocab_size).to(self.device)
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
            output.scatter_add_(1, encoder_word_idx.to(self.device), prob_ptr * ptr_output)
            if log_prob: output = torch.log(output + VERY_SMALL_NUMBER)
        else:
            if log_prob: output = F.log_softmax(logits, dim=1)
            else: output = F.softmax(logits, dim=1)

        return output, rnn_state, enc_attn, prob_ptr, enc_context

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

class Attention(nn.Module):
    def __init__(self, hidden_size, h_state_embed_size=None, in_memory_embed_size=None, attn_type='simple'):
        super(Attention, self).__init__()
        self.attn_type = attn_type
        if not h_state_embed_size:
            h_state_embed_size = hidden_size
        if not in_memory_embed_size:
            in_memory_embed_size = hidden_size
        if attn_type in ('mul', 'add'):
            self.W = torch.Tensor(h_state_embed_size, hidden_size)
            self.W = nn.Parameter(nn.init.xavier_uniform_(self.W))
            if attn_type == 'add':
                self.W2 = torch.Tensor(in_memory_embed_size, hidden_size)
                self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))
                self.W3 = torch.Tensor(hidden_size, 1)
                self.W3 = nn.Parameter(nn.init.xavier_uniform_(self.W3))
        elif attn_type == 'simple':
            pass
        else:
            raise RuntimeError('Unknown attn_type: {}'.format(self.attn_type))

    def forward(self, query_embed, in_memory_embed, attn_mask=None, addition_vec=None):
        if self.attn_type == 'simple': # simple attention
            attention = torch.bmm(in_memory_embed, query_embed.unsqueeze(2)).squeeze(2)
            if addition_vec is not None:
                attention = attention + addition_vec
        elif self.attn_type == 'mul': # multiplicative attention
            attention = torch.bmm(in_memory_embed, torch.mm(query_embed, self.W).unsqueeze(2)).squeeze(2)
            if addition_vec is not None:
                attention = attention + addition_vec
        elif self.attn_type == 'add': # additive attention
            attention = torch.mm(in_memory_embed.view(-1, in_memory_embed.size(-1)), self.W2)\
                .view(in_memory_embed.size(0), -1, self.W2.size(-1)) + torch.mm(query_embed, self.W).unsqueeze(1)
            if addition_vec is not None:
                attention = attention + addition_vec
            attention = torch.tanh(attention)
            attention = torch.mm(attention.view(-1, attention.size(-1)), self.W3).view(attention.size(0), -1)
        else:
            raise RuntimeError('Unknown attn_type: {}'.format(self.attn_type))

        if attn_mask is not None:
            # Exclude masked elements from the softmax
            attention = attn_mask * attention - (1 - attn_mask) * 1e20
        return attention

def create_mask(x, N, device=None):
    x = x.data
    mask = np.zeros((x.size(0), N))
    for i in range(x.size(0)):
        mask[i, :x[i]] = 1
    return torch.Tensor(mask).to(device)

class Dec_LSTM(nn.Module):
    '''
    Decoder LSTM cell with parent hidden state
    '''

    def __init__(self, emb_size, hidden_size, dropout, use_sibling):
        super(Dec_LSTM, self).__init__()
        self.rnn_size = hidden_size
        self.word_embedding_size = emb_size
        self.use_sibling = use_sibling

        if use_sibling:
            self.i2h = nn.Linear(self.word_embedding_size + 2*
                             self.rnn_size, 4*self.rnn_size)
        else:
            self.i2h = nn.Linear(self.word_embedding_size +
                             self.rnn_size, 4*self.rnn_size)

        self.h2h = nn.Linear(self.rnn_size, 4*self.rnn_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev_c, prev_h, parent_h, sibling_state):
        if self.use_sibling:
            input_cat = torch.cat((x, parent_h, sibling_state), 1)
        else:
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
    def __init__(self, input_size, emb_size, hidden_size, dropout_input, dropout_output, use_sibling):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_embedding_size = emb_size
        self.embedding = nn.Embedding(
            input_size, self.word_embedding_size, padding_idx=0)

        self.lstm = Dec_LSTM(emb_size, hidden_size, dropout_output, use_sibling)
        self.dropout = nn.Dropout(dropout_input)

    def forward(self, input_src, prev_c, prev_h, parent_h, sibling_state):

        src_emb = self.embedding(input_src)
        src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(
            src_emb, prev_c, prev_h, parent_h, sibling_state)
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
