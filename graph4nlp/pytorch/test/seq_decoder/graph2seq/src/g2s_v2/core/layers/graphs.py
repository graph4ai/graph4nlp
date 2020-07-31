import torch
import torch.nn as nn
import torch.nn.functional as F

from graph4nlp.pytorch.test.seq_decoder.graph2seq.src.g2s_v2.core.utils.generic_utils import to_cuda
from graph4nlp.pytorch.test.seq_decoder.graph2seq.src.g2s_v2.core.layers.common import GatedFusion, GRUStep


class GraphNN(nn.Module):
    def __init__(self, config):
        super(GraphNN, self).__init__()
        print('[ Using {}-hop GraphNN ]'.format(config['graph_hops']))
        self.device = config['device']
        hidden_size = config['hidden_size']
        self.graph_hops = config['graph_hops']
        self.linear_max = nn.Linear(hidden_size, hidden_size, bias=False)

        # Static graph
        self.graph_mp = GraphMessagePassing(config)
        self.gated_fusion = GatedFusion(hidden_size)
        self.gru_step = GRUStep(hidden_size, hidden_size)

    def forward(self, node_state, adj, node_mask=None):
        '''Static graph update'''
        node2edge, edge2node = adj

        # Shape: (batch_size, num_edges, num_nodes)
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        # Shape: (batch_size, num_nodes, num_edges)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)

        for _ in range(self.graph_hops):
            bw_agg_state = self.graph_mp.mp_func(node_state, node2edge, edge2node)
            fw_agg_state = self.graph_mp.mp_func(node_state, edge2node.transpose(1, 2), node2edge.transpose(1, 2))
            agg_state = self.gated_fusion(fw_agg_state, bw_agg_state)
            node_state = self.gru_step(node_state, agg_state)

        graph_embedding = self.graph_maxpool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding


    def graph_maxpool(self, node_state, node_mask=None):
        # Maxpool
        # Shape: (batch_size, hidden_size, num_nodes)
        node_embedding_p = self.linear_max(node_state).transpose(-1, -2)
        graph_embedding = F.max_pool1d(node_embedding_p, kernel_size=node_embedding_p.size(-1)).squeeze(-1)
        return graph_embedding


class GraphMessagePassing(nn.Module):
    def __init__(self, config):
        super(GraphMessagePassing, self).__init__()
        self.config = config
        hidden_size = config['hidden_size']
        self.mp_func = self.msg_pass

    def msg_pass(self, node_state, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size

        # Add self-loop
        norm_ = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, node2edge_emb) + node_state) / norm_
        return agg_state
