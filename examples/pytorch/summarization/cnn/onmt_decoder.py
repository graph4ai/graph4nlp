import torch
import torch.nn as nn

from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.modules import context_gate_factory, GlobalAttention
from onmt.utils.rnn_factory import rnn_factory

from onmt.utils.misc import aeq


class DecoderBase(nn.Module):
    """Abstract class for decoders.

    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor.

        Subclasses should override this method.
        """

        raise NotImplementedError


class RNNDecoderBase(DecoderBase):

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="mlp", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, copy_attn_type="general"):
        super(RNNDecoderBase, self).__init__(
            attentional=attn_type != "none" and attn_type is not None)

        self.max_step = 200

        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Decoder state
        self.state = {}

        # Build the RNN.

        self.out_prj = nn.Linear(hidden_size, self.embeddings.emb_luts[0].num_embeddings)

        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        # self.context_gate = None
        if context_gate is not None:
            self.context_gate = context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        if not self.attentional:
            if self._coverage:
                raise ValueError("Cannot use coverage term with no attention.")
            self.attn = None
        else:
            self.attn = GlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type, attn_func=attn_func
            )

        if copy_attn and not reuse_copy_attn:
            if copy_attn_type == "none" or copy_attn_type is None:
                raise ValueError(
                    "Cannot use copy_attn with copy_attn_type none")
            self.copy_attn = GlobalAttention(
                hidden_size, attn_type=copy_attn_type, attn_func=attn_func
            )
        else:
            self.copy_attn = None

        self._reuse_copy_attn = reuse_copy_attn and copy_attn
        if self._reuse_copy_attn and not self.attentional:
            raise ValueError("Cannot reuse copy attention with no attention.")

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout[0] if type(opt.dropout) is list
            else opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            opt.copy_attn_type)

    def init_state(self, src, memory_bank, encoder_final):
        """Initialize decoder state with last state of the encoder."""
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final)
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = fn(self.state["coverage"], 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, memory_bank, tgt=None, memory_lengths=None, step=None,
                **kwargs):

        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt=tgt, memory_bank=memory_bank, memory_lengths=memory_lengths)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        # self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        return dec_outs, attns

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.embeddings.update_dropout(dropout)


class InputFeedRNNDecoder(RNNDecoderBase):
    def _run_forward_pass(self, tgt=None, memory_bank=None, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        if tgt is not None:
            _, tgt_batch, _ = tgt.size()
            aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []
        if self.copy_attn is not None or self._reuse_copy_attn:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []


        # emb = self.embeddings(tgt)
        # assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.

        # for emb_t in emb.split(1):
        step_num = self.max_step
        if tgt is not None:
            step_num = min(step_num, tgt.shape[0]) + 1
            bos = torch.zeros(1, memory_bank.shape[1], 1).fill_(1).to(memory_bank.device).long()
            tgt = torch.cat((bos, tgt), dim=0)

            emb = self.embeddings(tgt)
        else:
            emb = None
            input_step = torch.zeros(1, memory_bank.shape[1], 1).fill_(1).to(memory_bank.device).long()


        for i in range(step_num):
            # emb_t = self.embeddings(input_step)
            if emb is not None:
                emb_t = emb[i]
            else:
                emb_t = self.embeddings(input_step)
            # decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            # print(decoder_input.shape)
            # exit(0)
            decoder_input = emb_t.squeeze(0)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            if self.attentional:
                decoder_output, p_attn = self.attn(
                    rnn_output,
                    memory_bank.transpose(0, 1),
                    memory_lengths=memory_lengths)
                attns["std"].append(p_attn)
            else:
                decoder_output = rnn_output
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output
            prob = self.out_prj(decoder_output)

            dec_outs += [prob.unsqueeze(1)]

            if tgt is not None:
                input_step = tgt[i, :, :]
            else:
                input_step = prob.argmax(1)
            input_step = input_step.view(1, memory_bank.shape[1], 1)

            # Update the coverage attention.
            if self._coverage:
                coverage = p_attn if coverage is None else p_attn + coverage
                attns["coverage"] += [coverage]

            if self.copy_attn is not None:
                _, copy_attn = self.copy_attn(
                    decoder_output, memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._reuse_copy_attn:
                attns["copy"] = attns["std"]
        dec_outs = torch.cat(dec_outs[:-1], dim=1)
        # print(dec_outs.shape)
        # exit(0)

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert rnn_type != "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)

#
# class InputFeedRNNDecoder(RNNDecoderBase):
#     """Input feeding based decoder.
#
#     See :class:`~onmt.decoders.decoder.RNNDecoderBase` for options.
#
#     Based around the input feeding approach from
#     "Effective Approaches to Attention-based Neural Machine Translation"
#     :cite:`Luong2015`
#
#
#     .. mermaid::
#
#        graph BT
#           A[Input n-1]
#           AB[Input n]
#           subgraph RNN
#             E[Pos n-1]
#             F[Pos n]
#             E --> F
#           end
#           G[Encoder]
#           H[memory_bank n-1]
#           A --> E
#           AB --> F
#           E --> H
#           G --> H
#     """
#
#     def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
#         """
#         See StdRNNDecoder._run_forward_pass() for description
#         of arguments and return values.
#         """
#         # Additional args check.
#         input_feed = self.state["input_feed"].squeeze(0)
#         input_feed_batch, _ = input_feed.size()
#         _, tgt_batch, _ = tgt.size()
#         aeq(tgt_batch, input_feed_batch)
#         # END Additional args check.
#
#         dec_outs = []
#         attns = {}
#         if self.attn is not None:
#             attns["std"] = []
#         if self.copy_attn is not None or self._reuse_copy_attn:
#             attns["copy"] = []
#         if self._coverage:
#             attns["coverage"] = []
#
#         bos = torch.zeros(1, memory_bank.shape[1], 1).fill_(1).to(memory_bank.device).long()
#         tgt = torch.cat((bos, tgt), dim=0)
#
#
#         emb = self.embeddings(tgt)
#         assert emb.dim() == 3  # len x batch x embedding_dim
#
#         dec_state = self.state["hidden"]
#         coverage = self.state["coverage"].squeeze(0) \
#             if self.state["coverage"] is not None else None
#
#         # Input feed concatenates hidden state with
#         # input at every time step.
#         for emb_t in emb.split(1):
#             decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
#             rnn_output, dec_state = self.rnn(decoder_input, dec_state)
#             if self.attentional:
#                 decoder_output, p_attn = self.attn(
#                     rnn_output,
#                     memory_bank.transpose(0, 1),
#                     memory_lengths=memory_lengths)
#                 attns["std"].append(p_attn)
#             else:
#                 decoder_output = rnn_output
#             if self.context_gate is not None:
#                 # TODO: context gate should be employed
#                 # instead of second RNN transform.
#                 decoder_output = self.context_gate(
#                     decoder_input, rnn_output, decoder_output
#                 )
#             decoder_output = self.dropout(decoder_output)
#             input_feed = decoder_output
#
#             dec_outs += [decoder_output.unsqueeze(1)]
#
#             # Update the coverage attention.
#             if self._coverage:
#                 coverage = p_attn if coverage is None else p_attn + coverage
#                 attns["coverage"] += [coverage]
#
#             if self.copy_attn is not None:
#                 _, copy_attn = self.copy_attn(
#                     decoder_output, memory_bank.transpose(0, 1))
#                 attns["copy"] += [copy_attn]
#             elif self._reuse_copy_attn:
#                 attns["copy"] = attns["std"]
#         dec_outs = torch.cat(dec_outs[:-1], dim=1)
#         dec_outs = self.out_prj(dec_outs)
#
#         return dec_state, dec_outs, attns
#
#     def _build_rnn(self, rnn_type, input_size,
#                    hidden_size, num_layers, dropout):
#         assert rnn_type != "SRU", "SRU doesn't support input feed! " \
#             "Please set -input_feed 0!"
#         stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
#         return stacked_cell(num_layers, input_size, hidden_size, dropout)
#
#     @property
#     def _input_size(self):
#         """Using input feed by concatenating input with attention vectors."""
#         return self.embeddings.embedding_size + self.hidden_size
#
#     def update_dropout(self, dropout):
#         self.dropout.p = dropout
#         self.rnn.dropout.p = dropout
#         self.embeddings.update_dropout(dropout)