import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, query_size, memory_size, hidden_size, has_bias=False, dropout=0.2, attention_funtion="mlp"):
        super(Attention, self).__init__()
        self.query_size = query_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.attn_type = attention_funtion
        self.dropout = nn.Dropout(dropout)
        assert self.attn_type in ["mlp", "general", "dot"]
        if self.attn_type == "mlp":
            self.query_in = nn.Linear(self.query_size, self.hidden_size, bias=True if has_bias else False)
            self.memory_in = nn.Linear(self.memory_size, self.hidden_size, bias=False)
        elif self.attn_type == "dot":
            if memory_size != query_size:
                raise ValueError("Parameter \"memory_size\" must be equal to \"query_size\"")
        elif self.attn_type == "general":
            self.query2memory = nn.Linear(query_size, memory_size, bias=False)
        self.out = nn.Linear(self.hidden_size, 1, bias=False)
        self.inf = 1e8

    def forward(self, query, memory, memory_mask=None, coverage=None):
        """
            Attention function
        Parameters
        ----------
        query: torch.Tensor, shape=[B, D]
        memory: torch.Tensor, shape=[B, N, D]
        memory_mask: torch.Tensor, shape=[B, N]
        coverage: torch.Tensor, shape=[B, N, D]

        Returns
        -------
        attn_results: torch.Tensor, shape=[B, D]
        attn_scores: torch.Tensor, shape=[B, N]
        """
        assert len(query.shape) == 2
        assert len(memory.shape) == 3
        assert query.shape[0] == memory.shape[0]

        aligns = self._calculate_aligns(query, memory, coverage=coverage)

        if memory_mask is not None:
            aligns = aligns.masked_fill(memory_mask == 0, -self.inf)
        scores = torch.softmax(aligns, dim=-1)
        scores = self.dropout(scores)
        ret = torch.bmm(scores.unsqueeze(1), memory)
        return ret.squeeze(1), scores

    def _calculate_aligns(self, src, tgt, coverage=None):
        """
            Attention score calculation.
        Parameters
        ----------
        src: torch.Tensor, shape=[B, D]
        tgt: torch.Tensor, shape=[B, N, D]
        coverage: torch.Tensor, shape=[B, N, D]
        Returns
        -------
        aligns: torch.Tensor, shape=[B, N]
        """
        if self.attn_type == "mlp":
            src = self.query_in(src)
            tgt = self.memory_in(tgt)
            aligns = src.unsqueeze(1) + tgt
            if coverage is not None:
                assert len(coverage.shape) == 3
                assert coverage.shape[-1] == self.hidden_size
                aligns += coverage
            aligns = torch.tanh(aligns)
            aligns = self.out(aligns)
            aligns = aligns.view(aligns.shape[0], aligns.shape[1])
        elif self.attn_type == "general" or self.attn_type == "dot":
            if coverage:
                assert tgt.shape == coverage.shape
                tgt = tgt + coverage
                tgt = torch.tanh(tgt)

            if self.attn_type == "general":
                src = self.query2memory(src)
            aligns = torch.bmm(src.unsqueeze(1), tgt.transpose(1, 2))
            aligns = aligns.squeeze(1).contiguous()
        else:
            raise NotImplementedError()
        return aligns

class AttnUnit(nn.Module):
    def __init__(self, hidden_size, output_size, attention_type, dropout):
        super(AttnUnit, self).__init__()
        self.hidden_size = hidden_size
        self.separate_attention = (attention_type != "uniform")

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

        h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)

        return pred
