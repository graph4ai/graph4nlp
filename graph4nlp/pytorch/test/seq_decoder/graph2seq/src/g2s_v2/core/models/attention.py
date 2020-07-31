import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size, attn_type="add"):
        super(Attention, self).__init__()


    def forward(self, query, memory, memory_mask=None, addictive_vec=None):
        """
            Attention function
        Parameters
        ----------
        query: torch.Tensor, shape=[B, ]
        memory
        memory_mask
        addictive_vec

        Returns
        -------

        """
        pass