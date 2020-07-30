import argparse
import os
import pickle as pkl
from sys import path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from . import data_utils


def generate_embedding_from_glove(args):
    data_dir = args.vocab_data_dir
    min_freq = 2
    max_vocab_size = 15000
    pretrained_embedding_dir = args.pretrained_embedding_text

    word_manager = data_utils.SymbolsManager(True)
    word_manager.init_from_file(
        "{}/vocab.q.txt".format(data_dir), min_freq, max_vocab_size)

    glove2vec = {}
    words_arr = []
    cnt_find = 0
    with open(pretrained_embedding_dir, "r") as f:
        for l in tqdm(f):
            line = l.split()
            word = line[0]
            words_arr.append(word)
            vect = np.array(line[1:]).astype(np.float)
            glove2vec[word] = vect

    word2vec = {}
    word_arr = list(word_manager.symbol2idx.keys())
    for w in tqdm(word_arr):
        if w in list(glove2vec.keys()):
            word2vec[w] = glove2vec[w]

    print(len(word2vec))
    out_file = "{}/pretrain.pkl".format(data_dir)
    with open(out_file, "wb") as out_data:
        pkl.dump(word2vec, out_data)
