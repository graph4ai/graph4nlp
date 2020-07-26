'''This models is an example for training a classifier on SNLI'''
from __future__ import print_function
from os.path import join

import nltk
import numpy as np
import os
import urllib
import zipfile
import sys

from spodernet.hooks import AccuracyHook, LossHook, ETAHook
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import AddToVocab, CreateBinsByNestedLength, SaveLengthsToState, ConvertTokenToIdx, StreamToHDF5, Tokenizer, NaiveNCharTokenizer
from spodernet.preprocessing.processors import JsonLoaderProcessors, DictKey2ListMapper, RemoveLineOnJsonValueCondition, ToLower
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.utils.logger import Logger, LogLevel
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.util import get_data_path

from torch.nn.modules.rnn import LSTM
from torch.autograd import Variable
import torch

Config.parse_argv(sys.argv)

np.set_printoptions(suppress=True)

class Net(torch.nn.Module):

    def __init__(self, num_embeddings, num_labels):
        super(Net, self).__init__()
        self.emb = torch.nn.Embedding(num_embeddings, Config.embedding_dim, padding_idx=0)
        self.lstm1 = LSTM(Config.embedding_dim, Config.hidden_size, num_layers=1, batch_first=True, bias=True, dropout=Config.dropout, bidirectional=True)
        self.lstm2 = LSTM(Config.embedding_dim, Config.hidden_size, num_layers=1, batch_first=True, bias=True, dropout=Config.dropout, bidirectional=True)
        self.linear = torch.nn.Linear(Config.hidden_size*4, num_labels)
        self.loss = torch.nn.CrossEntropyLoss()
        self.pred = torch.nn.Softmax()

        self.h0 = Variable(torch.zeros(2,Config.batch_size, Config.hidden_size))
        self.c0 = Variable(torch.zeros(2,Config.batch_size, Config.hidden_size))
        self.h1 = Variable(torch.zeros(2,Config.batch_size, Config.hidden_size))
        self.c1 = Variable(torch.zeros(2,Config.batch_size, Config.hidden_size))

        if Config.cuda:
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()
            self.h1 = self.h1.cuda()
            self.c1 = self.c1.cuda()

    def forward(self, str2var):
        inp = str2var['input']
        sup = str2var['support']
        l1 = str2var['input_length']
        l2 = str2var['support_length']
        t = str2var['target']
        self.h0.data.zero_()
        self.c0.data.zero_()

        inp_seq = self.emb(inp)
        sup_seq = self.emb(sup)

        out1, hid1 = self.lstm1(inp_seq, (self.h0, self.c0))
        out2, hid2 = self.lstm2(sup_seq, hid1)

        outs1 = []
        outs2 = []
        for i in range(Config.batch_size):
            outs1.append(out1[i,l1.data[i]-1, :])
            outs2.append(out2[i,l2.data[i]-1, :])

        out1_stacked = torch.stack(outs1, 0)
        out2_stacked = torch.stack(outs2, 0)
        out = torch.cat([out1_stacked, out2_stacked], 1)
        projected = self.linear(out)
        loss = self.loss(projected, t)
        max_values, argmax = torch.topk(self.pred(projected),1)
        return loss, argmax


def download_snli():
    '''Creates data and snli paths and downloads SNLI in the home dir'''
    home = os.environ['HOME']
    data_dir = join(home, '.data')
    snli_dir = join(data_dir, 'snli')
    snli_url = 'http://nlp.stanford.edu/projects/snli/snli_1.0.zip'

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(snli_dir):
        os.mkdir(snli_dir)

    if not os.path.exists(join(data_dir, 'snli_1.0.zip')):
        print('Downloading SNLI...')
        snlidownload = urllib.URLopener()
        snlidownload.retrieve(snli_url, join(data_dir, "snli_1.0.zip"))

    print('Opening zip file...')
    archive = zipfile.ZipFile(join(data_dir, 'snli_1.0.zip'), 'r')

    return archive, snli_dir


def snli2json():
    '''Preprocesses SNLI data and returns to spoder files'''
    files = ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl',
             'snli_1.0_test.jsonl']

    archive, snli_dir = download_snli()

    new_files = ['train.data', 'dev.data', 'test.data']
    names = ['train', 'dev', 'test']

    if not os.path.exists(join(snli_dir, new_files[0])):
        for name, new_name in zip(files, new_files):
            print('Writing {0}...'.format(new_name))
            archive = zipfile.ZipFile(join(data_dir, 'snli_1.0.zip'), 'r')
            snli_file = archive.open(join('snli_1.0', name), 'r')
            with open(join(snli_dir, new_name), 'w') as datafile:
                for line in snli_file:
                    data = json.loads((line))
                    if data['gold_label'] == '-':
                        continue

                    premise = data['sentence1']
                    hypothesis = data['sentence2']
                    target = data['gold_label']
                    datafile.write(
                        json.dumps([premise, hypothesis, target]) + '\n')

    return [names, [join(snli_dir, new_name) for new_name in new_files]]

def preprocess_SNLI(delete_data=False):
    # load data
    #names, file_paths = snli2json()
    #train_path, dev_path, test_path = file_paths
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    zip_path = join(get_data_path(), 'snli_1.0.zip', 'snli_1.0')
    file_paths = ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl']

    not_t = []
    t = ['input', 'support', 'target']
    # tokenize and convert to hdf5
    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline('snli_example', delete_data)
    p.add_path(join(zip_path, file_paths[0]))
    p.add_line_processor(JsonLoaderProcessors())
    p.add_line_processor(RemoveLineOnJsonValueCondition('gold_label', lambda label: label == '-'))
    p.add_line_processor(DictKey2ListMapper(['sentence1', 'sentence2', 'gold_label']))
    p.add_sent_processor(ToLower())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p.add_token_processor(AddToVocab())
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()
    p.state['vocab'].save_to_disk()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(ToLower())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(CreateBinsByNestedLength('snli_train', min_batch_size=128))
    state = p.execute()

    # dev and test data
    p2 = Pipeline('snli_example')
    p2.copy_vocab_from_pipeline(p)
    p2.add_path(join(zip_path, file_paths[1]))
    p2.add_line_processor(JsonLoaderProcessors())
    p2.add_line_processor(RemoveLineOnJsonValueCondition('gold_label', lambda label: label == '-'))
    p2.add_line_processor(DictKey2ListMapper(['sentence1', 'sentence2', 'gold_label']))
    p2.add_sent_processor(ToLower())
    p2.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p2.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p2.add_post_processor(SaveLengthsToState())
    p2.execute()

    p2.clear_processors()
    p2.add_sent_processor(ToLower())
    p2.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p2.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p2.add_post_processor(ConvertTokenToIdx())
    p2.add_post_processor(StreamToHDF5('snli_dev'))
    p2.execute()

    p3 = Pipeline('snli_example')
    p3.copy_vocab_from_pipeline(p)
    p3.add_path(join(zip_path, file_paths[2]))
    p3.add_line_processor(JsonLoaderProcessors())
    p3.add_line_processor(RemoveLineOnJsonValueCondition('gold_label', lambda label: label == '-'))
    p3.add_line_processor(DictKey2ListMapper(['sentence1', 'sentence2', 'gold_label']))
    p3.add_sent_processor(ToLower())
    p3.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p3.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p3.add_post_processor(SaveLengthsToState())
    p3.execute()

    p3.clear_processors()
    p3.add_sent_processor(ToLower())
    p3.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p3.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p3.add_post_processor(ConvertTokenToIdx())
    p3.add_post_processor(StreamToHDF5('snli_test'))
    p3.execute()



def main():
    Logger.GLOBAL_LOG_LEVEL = LogLevel.INFO
    #Config.backend = Backends.TENSORFLOW
    Config.backend = Backends.TORCH
    Config.cuda = True
    Config.dropout = 0.1
    Config.hidden_size = 128
    Config.embedding_size = 256
    Config.L2 = 0.00003

    do_process = False
    if do_process:
        preprocess_SNLI(delete_data=True)


    p = Pipeline('snli_example')
    vocab = p.state['vocab']
    vocab.load_from_disk()

    batch_size = 128
    if Config.backend == Backends.TENSORFLOW:
        from spodernet.backends.tfbackend import TensorFlowConfig
        TensorFlowConfig.init_batch_size(batch_size)
    train_batcher = StreamBatcher('snli_example', 'snli_train', batch_size, randomize=True, loader_threads=8)
    #train_batcher.subscribe_to_batch_prepared_event(SomeExpensivePreprocessing())
    dev_batcher = StreamBatcher('snli_example', 'snli_dev', batch_size)
    test_batcher  = StreamBatcher('snli_example', 'snli_test', batch_size)

    #train_batcher.subscribe_to_events(AccuracyHook('Train', print_every_x_batches=1000))
    train_batcher.subscribe_to_events(LossHook('Train', print_every_x_batches=100))
    train_batcher.subscribe_to_events(AccuracyHook('Train', print_every_x_batches=100))
    dev_batcher.subscribe_to_events(AccuracyHook('Dev', print_every_x_batches=100))
    dev_batcher.subscribe_to_events(LossHook('Dev', print_every_x_batches=100))
    eta = ETAHook(print_every_x_batches=100)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)

    net = Net(vocab.num_embeddings, vocab.num_labels)
    if Config.cuda:
        net.cuda()

    epochs = 10
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for epoch in range(epochs):
        for str2var in train_batcher:
            opt.zero_grad()
            loss, argmax = net(str2var)
            loss.backward()
            opt.step()
            train_batcher.state.loss = loss
            train_batcher.state.targets = str2var['target']
            train_batcher.state.argmax = argmax

    net.eval()
    for i, str2var in enumerate(dev_batcher):
        t = str2var['target']
        idx = str2var['index']
        loss, argmax = net(str2var)
        dev_batcher.state.loss = loss
        dev_batcher.state.targets = str2var['target']
        dev_batcher.state.argmax = argmax


if __name__ == '__main__':
    main()
