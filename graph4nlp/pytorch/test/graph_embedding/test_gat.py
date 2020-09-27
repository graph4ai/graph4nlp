import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh

from ...modules.graph_embedding.gat import GAT


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    graph = DGLGraph(data.graph)
    return graph, features, labels, mask

class GNNClassifier(nn.Module):
    def __init__(self,
                num_layers,
                input_size,
                hidden_size,
                output_size,
                num_heads,
                num_out_heads,
                direction_option):
        super(GNNClassifier, self).__init__()
        self.direction_option = direction_option
        heads = [num_heads] * (num_layers - 1) + [num_out_heads]
        self.model = GAT(num_layers,
                    input_size,
                    hidden_size,
                    output_size,
                    heads,
                    direction_option=direction_option,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=0.2,
                    residual=False,
                    activation=F.elu)

        if self.direction_option == 'bi_sep':
            self.fc = nn.Linear(2 * output_size, output_size)

    def forward(self, graph, features):
        logits = self.model(graph, features)
        if self.direction_option == 'bi_sep':
            logits = self.fc(F.elu(logits))

        return logits

if __name__ == '__main__':
    graph, features, labels, mask = load_cora_data()

    num_layers = 2
    input_size = features.size()[1]
    hidden_size = 8
    output_size = 7
    num_heads = 2
    num_out_heads = 1
    direction_option = 'uni' # 'uni', 'bi_sep', 'bi_fuse'
    num_epochs = 30

    classifier = GNNClassifier(num_layers,
            input_size,
            hidden_size,
            output_size,
            num_heads,
            num_out_heads,
            direction_option)

    # create optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    # main loop
    dur = []
    for epoch in range(num_epochs):
        t0 = time.time()
        logits = classifier(graph, features)
        assert logits.shape[-1] == output_size

        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)

        print("Epoch {} | Loss {:.4f} | Time(s) {:.2f}".format(
            epoch, loss.item(), np.mean(dur)))
