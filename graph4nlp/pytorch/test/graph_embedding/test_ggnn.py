import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh

from ...modules.graph_embedding.ggnn import GGNN
# from graph4nlp.pytorch.modules.graph_embedding.ggnn import GGNN

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
                output_size,
                direction_option):
        super(GNNClassifier, self).__init__()
        self.direction_option = direction_option
        self.model = GGNN(num_layers, input_size, output_size, direction_option=direction_option)

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
    print(input_size)
    output_size = features.size()[1]
    direction_option = 'bi_sep' # 'uni', 'bi_sep', 'bi_fuse'
    num_epochs = 30

    classifier = GNNClassifier(num_layers,
            input_size,
            output_size,
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

# from dgl.nn.pytorch import GatedGraphConv
# import torch.nn as nn
# import torch.nn.functional as F
# import dgl
# import torch
# from dgl.data import MiniGCDataset
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import pickle as pkl
# from ...modules.graph_embedding.ggnn import GGNN
# import os
#
# dgl.random.seed(123)
# torch.manual_seed(123)
# torch.cuda.manual_seed(123)
# torch.backends.cudnn.deterministic = True
#
# def collate(samples):
#     # The input `samples` is a list of pairs
#     #  (graph, label).
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels)
#
#
# class Classifier(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes, direction_option):
#         super(Classifier, self).__init__()
#
#         if direction_option=='bi_fuse':
#             self.encoder = GGNN(2, in_dim, hidden_dim, direction_option='bi_fuse')
#             self.classify = nn.Linear(hidden_dim, n_classes)
#         elif direction_option=='bi_sep':
#             self.encoder = GGNN(2, in_dim, hidden_dim, direction_option='bi_sep')
#             self.classify = nn.Linear(hidden_dim * 2, n_classes)
#         else:
#             self.encoder = GGNN(2, in_dim, hidden_dim, direction_option='uni')
#             self.classify = nn.Linear(hidden_dim, n_classes)
#
#
#     def forward(self, g, node_feats):
#         h = self.encoder(g, node_feats)
#         g.ndata['h'] = h
#         # Calculate graph representation by averaging all the node representations.
#         hg = dgl.mean_nodes(g, 'h')
#         return self.classify(hg)
#
#
# if __name__ == '__main__':
#     # Create training and test sets.
#     # trainset = MiniGCDataset(320, 10, 20)
#     # testset = MiniGCDataset(80, 10, 20)
#     # Use PyTorch's DataLoader and the collate function
#     # defined before.
#
#     # with open('train.pkl','wb') as f:
#     #     pkl.dump(trainset, f)
#     #
#     # with open('test.pkl','wb') as f:
#     #     pkl.dump(testset, f)
#
#     with open('graph4nlp/pytorch/test/graph_embedding/train.pkl','rb') as f:
#         trainset = pkl.load(f)
#
#     with open('graph4nlp/pytorch/test/graph_embedding/test.pkl','rb') as f:
#         testset = pkl.load(f)
#
#     data_loader = DataLoader(trainset, batch_size=8, shuffle=True,
#                              collate_fn=collate)
#
#     # Create model
#     model = Classifier(1, 256, trainset.num_classes, direction_option='uni')
#     loss_func = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     model.train()
#
#     epoch_losses = []
#     for epoch in range(30):
#         epoch_loss = 0
#         for iter, (bg, label) in enumerate(data_loader):
#             node_feats = bg.in_degrees().view(-1, 1).float()
#             prediction = model(bg, node_feats)
#             loss = loss_func(prediction, label)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.detach().item()
#         epoch_loss /= (iter + 1)
#         print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
#         epoch_losses.append(epoch_loss)
#
#     model.eval()
#     # Convert a list of tuples to two lists
#     test_X, test_Y = map(list, zip(*testset))
#     test_bg = dgl.batch(test_X)
#     test_Y = torch.tensor(test_Y).float().view(-1, 1)
#     test_node_feats = test_bg.in_degrees().view(-1, 1).float()
#     probs_Y = torch.softmax(model(test_bg, test_node_feats), 1)
#     sampled_Y = torch.multinomial(probs_Y, 1)
#     argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
#     print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
#         (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
#     print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
#         (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))