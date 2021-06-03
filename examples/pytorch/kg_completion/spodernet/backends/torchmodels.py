from torch.nn import LSTM
from torch.autograd import Variable

import torch
import torch.nn.functional as F

from spodernet.frontend import AbstractModel
from spodernet.utils.global_config import Config

class TorchEmbedding(torch.nn.Module, AbstractModel):
    def __init__(self, embedding_size, num_embeddings):
        super(TorchEmbedding, self).__init__()

        self.emb= torch.nn.Embedding(num_embeddings,
                embedding_size, padding_idx=0)#, scale_grad_by_freq=True, padding_idx=0)

    def forward(self, str2var, *args):
        self.expected_str2var_keys_oneof(str2var, ['input', 'support'])
        self.expected_args('None', 'None')
        self.generated_outputs('input idx, support idx', 'both sequences have shape = [batch, timesteps, embedding dim]')

        embedded_results = []
        if 'input' in str2var:
            embedded_results.append(self.emb(str2var['input']))

        if 'support' in str2var:
            embedded_results.append(self.emb(str2var['support']))

        return embedded_results

class TorchBiDirectionalLSTM(torch.nn.Module, AbstractModel):
    def __init__(self, input_size, hidden_size,
            dropout=0.0, layers=1,
            bidirectional=True, to_cuda=False, conditional_encoding=True):
        super(TorchBiDirectionalLSTM, self).__init__()

        use_bias = True
        num_directions = (1 if not bidirectional else 2)

        self.lstm = LSTM(input_size,hidden_size,layers,
                         use_bias,True,0.2,bidirectional)

        # states of both LSTMs
        self.h0 = None
        self.c0 = None

        self.h0 = Variable(torch.FloatTensor(num_directions*layers, Config.batch_size, hidden_size))
        self.c0 = Variable(torch.FloatTensor(num_directions*layers, Config.batch_size, hidden_size))

        if Config.cuda:
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()

    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, [])
        self.expected_args('embedded seq', 'size [batch, time steps, embedding dim]')
        self.generated_outputs('LSTM output seq', 'size [batch, time steps, 2x hidden dim]')
        seq = args
        self.h0.data.zero_()
        self.c0.data.zero_()
        out, hid = self.lstm(seq, (self.h0, self.c0))
        return [out, hid]


class TorchPairedBiDirectionalLSTM(torch.nn.Module, AbstractModel):
    def __init__(self, input_size, hidden_size,
            dropout=0.0, layers=1,
            bidirectional=True, to_cuda=False, conditional_encoding=True):
        super(TorchPairedBiDirectionalLSTM, self).__init__()

        self.conditional_encoding = conditional_encoding
        use_bias = True
        num_directions = (1 if not bidirectional else 2)

        self.conditional_encoding = conditional_encoding
        self.lstm1 = LSTM(input_size,hidden_size,layers,
                         use_bias,True,Config.dropout,bidirectional)
        self.lstm2 = LSTM(input_size,hidden_size,layers,
                         use_bias,True,Config.dropout,bidirectional)

        # states of both LSTMs
        self.h01 = None
        self.c01 = None
        self.h02 = None
        self.c02 = None


        self.h01 = Variable(torch.FloatTensor(num_directions*layers, Config.batch_size, hidden_size))
        self.c01 = Variable(torch.FloatTensor(num_directions*layers, Config.batch_size, hidden_size))

        if Config.cuda:
            self.h01 = self.h01.cuda()
            self.c01 = self.c01.cuda()

        if not self.conditional_encoding:
            self.h02 = Variable(torch.FloatTensor(num_directions*layers, Config.batch_size, hidden_size))
            self.c02 = Variable(torch.FloatTensor(num_directions*layers, Config.batch_size, hidden_size))

            if Config.cuda:
                self.h02 = self.h02.cuda()
                self.c02 = self.c02.cuda()


    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, [])
        self.expected_args('embedded input seq, embedded seq support', 'both of size [batch, time steps, embedding dim]')
        self.generated_outputs('LSTM output seq inputs, LSTM output seq support', 'both of size [batch, time steps, 2x hidden dim]')
        seq1, seq2 = args
        if self.conditional_encoding:
            self.h01.data.zero_()
            self.c01.data.zero_()
            out1, hid1 = self.lstm1(seq1, (self.h01, self.c01))
            out2, hid2 = self.lstm2(seq2, hid1)
        else:
            self.h01.data.zero_()
            self.c01.data.zero_()
            self.h02.data.zero_()
            self.c02.data.zero_()
            out1, hid1 = self.lstm1(seq1, (self.h01, self.c01))
            out2, hid2 = self.lstm2(seq2, (self.h02, self.c02))
        return [out1, out2]

class TorchVariableLengthOutputSelection(torch.nn.Module, AbstractModel):
    def __init__(self):
        super(TorchVariableLengthOutputSelection, self).__init__()
        self.b1 = None
        self.b2 = None

    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, ['input_length', 'support_length'])
        self.expected_args('LSTM output sequence input , LSTM output sequence support', 'dimension of both: [batch, time steps, 2x LSTM hidden size]')
        self.generated_outputs('stacked bidirectional outputs of last timestep', 'dim is [batch_size, 4x hidden size]')

        output_lstm1, output_lstm2 = args

        l1, l2 = str2var['input_length'], str2var['support_length']
        if self.b1 == None:
            b1 = torch.ByteTensor(output_lstm1.size())
            b2 = torch.ByteTensor(output_lstm2.size())
            if Config.cuda:
                b1 = b1.cuda()
                b2 = b2.cuda()

        b1.fill_(0)
        for i, num in enumerate(l1.data):
            b1[i,num-1,:] = 1
        out1 = output_lstm1[b1].view(Config.batch_size, -1)

        b2.fill_(0)
        for i, num in enumerate(l2.data):
            b2[i,num-1,:] = 1
        out2 = output_lstm2[b2].view(Config.batch_size, -1)

        out = torch.cat([out1,out2], 1)
        return [out]

class TorchSoftmaxCrossEntropy(torch.nn.Module, AbstractModel):

    def __init__(self, input_dim, num_labels):
        super(TorchSoftmaxCrossEntropy, self).__init__()
        self.num_labels = num_labels
        self.projection_to_labels = torch.nn.Linear(input_dim, num_labels)

    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, ['target'])
        self.expected_args('some inputs', 'dimension: [batch, any]')
        self.generated_outputs('logits, loss, argmax', 'dimensions: logits = [batch, labels], loss = 1x1, argmax = [batch, 1]')

        outputs_prev_layer = args[0]
        t = str2var['target']

        logits = self.projection_to_labels(outputs_prev_layer)
        out = F.log_softmax(logits)
        loss = F.nll_loss(out, t)
        maximum, argmax = torch.topk(out.data, 1)

        return [logits, loss, argmax]

