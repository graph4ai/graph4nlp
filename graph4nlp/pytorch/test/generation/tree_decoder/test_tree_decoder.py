import random
import warnings

import numpy as np
import torch
import time
from stanfordcorenlp import StanfordCoreNLP
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim

from ....data.data import GraphData
from ....modules.prediction.generation.TreeBasedDecoder import StdTreeDecoder
from ....modules.utils.tree_utils import DataLoader, Tree, Vocab, to_cuda

warnings.filterwarnings('ignore')

class Seq2Tree(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


class SequenceEncoder(nn.Module):
    def __init__(self, input_size, enc_emb_size, enc_hidden_size, dec_hidden_size, dropout_input, dropout_output, pad_idx=0, encode_rnn_num_layer=1):
        super(SequenceEncoder, self).__init__()
        self.embedding = nn.Embedding(
            input_size, enc_emb_size, padding_idx=pad_idx)
        self.rnn = nn.LSTM(enc_emb_size, enc_hidden_size, encode_rnn_num_layer,
                           bias=True, batch_first=True, dropout=dropout_output, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_size * 4, dec_hidden_size)
        self.dropout = None
        if dropout_input > 0:
            self.dropout = nn.Dropout(dropout_input)

    def forward(self, input_src):
        # batch_size x src_length x emb_size
        src_emb = self.dropout(self.embedding(input_src))
        # output: [batch size, src len, hid dim * num directions], hidden: a tuple of length "n layers * num directions", each element in tuple is
        output, (hn, cn) = self.rnn(src_emb)
        hidden = torch.tanh(self.fc(
            torch.cat((hn[-2, :, :], hn[-1, :, :], cn[-2, :, :], cn[-1, :, :]), dim=1)))
        return output, hidden


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


if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    '''loading data and begin training'''
    src_vocab_file = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\jobs640\vocab.q.txt"
    tgt_vocab_file = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\jobs640\vocab.f.txt"
    data_file = r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\jobs640\train.txt"
    mode = "train"
    min_freq = 2
    max_vocab_size = 10000
    batch_size = 20
    device = None

    train_data_loader = DataLoader(src_vocab_file=src_vocab_file, tgt_vocab_file=tgt_vocab_file, data_file=data_file,
                                   mode=mode, min_freq=min_freq, max_vocab_size=max_vocab_size, batch_size=batch_size, device=device)

    '''For encoder-decoder'''
    input_size = train_data_loader.src_vocab.vocab_size
    output_size = train_data_loader.tgt_vocab.vocab_size
    enc_emb_size = 300
    tgt_emb_size = 300
    enc_hidden_size = 300
    dec_hidden_size = 600
    enc_dropout_input = 0.1
    enc_dropout_output = 0.3
    dec_dropout_input = 0.1
    dec_dropout_output = 0.3
    attn_dropout = 0.1

    encoder = SequenceEncoder(input_size=input_size, enc_emb_size=enc_emb_size, enc_hidden_size=enc_hidden_size, dec_hidden_size=dec_hidden_size,
                              pad_idx=train_data_loader.src_vocab.get_symbol_idx(train_data_loader.src_vocab.pad_token), dropout_input=enc_dropout_input, dropout_output=enc_dropout_output, encode_rnn_num_layer=1)

    '''For attention unit'''
    attention_type = "uniform"
    criterion = nn.NLLLoss(size_average=False, ignore_index=0)
    
    attn_unit = AttnUnit(dec_hidden_size, output_size, attention_type, attn_dropout)
    tree_decoder = StdTreeDecoder(attn=attn_unit, dec_emb_size=tgt_emb_size, dec_hidden_size=dec_hidden_size, output_size=output_size, device=device, criterion=criterion, use_sibling=True, use_attention=True, use_copy=False, use_coverage=False,
                                  fuse_strategy="average", num_layers=1, dropout_input=dec_dropout_input, dropout_output=dec_dropout_output, rnn_type="lstm", max_dec_seq_length=512, max_dec_tree_depth=256, tgt_vocab=train_data_loader.tgt_vocab)

    to_cuda(encoder, device)
    to_cuda(tree_decoder, device)

    init_weight = 0.08
    print("encoder:")
    for name, param in encoder.named_parameters():
        # print(name,param.size())
        if param.requires_grad:
            # print(name,param.size())
            # print(name)
            if ("embedding.weight" in name) or ("bert_embedding" in name):
                print("Do not initialize pretrained embedding parameters")
            else:
                if len(param.size()) >= 2:
                    if "rnn" in name:
                        init.orthogonal_(param)
                    else:
                        init.xavier_uniform_(param, gain=1.0)
                else:
                    init.uniform_(param, -init_weight, init_weight)
                # init.xavier_uniform_(param, gain=1.0)

    print("decoder:")
    for name, param in tree_decoder.named_parameters():
        if param.requires_grad:
            # print(name,param.size())
            if "rnn" in name and len(param.size()) >= 2:
                init.orthogonal_(param)
            else:
                init.uniform_(param, -init_weight, init_weight)

    max_epochs = 300
    step = 0
    epoch = 0
    grad_clip = 5
    optim_state = {"learningRate": 1e-3,
                   "weight_decay":  1e-5}
    print("using adam")
    encoder_optimizer = optim.Adam(encoder.parameters(
    ),  lr=optim_state["learningRate"], weight_decay=optim_state["weight_decay"])

    decoder_optimizer = optim.Adam(
        tree_decoder.parameters(),  lr=optim_state["learningRate"])


    print("Starting training.")
    encoder.train()
    tree_decoder.train()

    iterations = max_epochs * train_data_loader.num_batch
    start_time = time.time()

    best_val_acc = 0

    print("Batch number per Epoch:", train_data_loader.num_batch)
    print_every = train_data_loader.num_batch

    loss_to_print = 0
    for i in range(iterations):
        if (i+1) % train_data_loader.num_batch == 0:
            epoch += 1

        epoch = i // train_data_loader.num_batch

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        enc_batch, enc_len_batch, dec_tree_batch = train_data_loader.random_batch()

        output_, hidden_ = encoder(enc_batch)
        encode_output_dict = {'graph_node_embedding': None, 'graph_node_mask':None, 'graph_edge_embedding':None, 'rnn_node_embedding':output_, 'graph_level_embedding':hidden_, 'graph_edge_mask':None}
        loss = tree_decoder(encode_output_dict, dec_tree_batch)
        loss.backward()
        torch.nn.utils.clip_grad_value_(encoder.parameters(),grad_clip)
        torch.nn.utils.clip_grad_value_(tree_decoder.parameters(),grad_clip)
        encoder_optimizer.step()
        decoder_optimizer.step()

        loss_to_print += loss
        if i == iterations - 1 or (i+1) % print_every == 0:
            pass
            # checkpoint = {}
            # checkpoint["encoder"] = encoder
            # checkpoint["decoder"] = decoder
            # checkpoint["attention_decoder"] = attention_decoder
            # checkpoint["opt"] = opt
            # checkpoint["i"] = i
            # checkpoint["epoch"] = epoch
            # if epoch % opt.save_when_train == 0:
            #     torch.save(
            #         checkpoint, "{}/model_g2t".format(opt.checkpoint_dir) + str(i))

        if (i+1) % print_every == 0:
            end_time = time.time()
            print(("{}/{}, train_loss = {}, epochs = {}, time since last print = {}".format(i,
                                                                                           iterations, (loss_to_print / print_every), epoch, (end_time - start_time)/60)))
            loss_to_print = 0
            start_time = time.time()

        if loss != loss:
            print('loss is NaN.  This usually indicates a bug.')
            break

