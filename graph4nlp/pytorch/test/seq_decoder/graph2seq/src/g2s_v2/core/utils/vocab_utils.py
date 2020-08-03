# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import re
import pickle
import numpy as np
from collections import Counter
from functools import lru_cache

from . import constants
from .data_utils import tokenize

word_detector = re.compile('\w')

class VocabModel(object):
    def __init__(self, data_set, config):
        print('Building vocabs...')
        (allWords, allEdgeTypes) = collect_vocabs(data_set)
        print('Number of words: {}'.format(len(allWords)))
        print('Number of edge types: {}'.format(len(allEdgeTypes)))

        self.word_vocab = Vocab()
        self.word_vocab.build_vocab(allWords, vocab_size=config['top_word_vocab'], min_freq=config['min_word_freq'])
        if config.get('pretrained_word_embed_file', None):
            self.word_vocab.load_embeddings(config['pretrained_word_embed_file'])
            print('Using pretrained word embeddings')
        else:
            self.word_vocab.randomize_embeddings(config['word_embed_dim'])
            print('Using randomized word embeddings')
        print('word_vocab: {}'.format(self.word_vocab.embeddings.shape))

        self.edge_vocab = Vocab()
        self.edge_vocab.build_vocab(allEdgeTypes)
        print('edge_vocab: {}'.format((self.edge_vocab.get_vocab_size())))


    @classmethod
    def build(cls, saved_vocab_file=None, data_set=None, config=None):
        """
        Loads a Vocabulary from disk.

        Args:
            saved_vocab_file (str): path to the saved vocab file
            data_set:
            config:

        Returns:
            Vocabulary: loaded Vocabulary
        """
        if os.path.exists(saved_vocab_file):
            print('Loading pre-built vocab model stored in {}'.format(saved_vocab_file))
            vocab_model = pickle.load(open(saved_vocab_file, 'rb'))

        else:
            vocab_model = VocabModel(data_set, config)
            print('Saving vocab model to {}'.format(saved_vocab_file))
            pickle.dump(vocab_model, open(saved_vocab_file, 'wb'))
        return vocab_model

class Vocab(object):
    def __init__(self):
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.UNK = 3
        self.pad_token = constants._PAD_TOKEN
        self.sos_token = constants._SOS_TOKEN
        self.eos_token = constants._EOS_TOKEN
        self.unk_token = constants._UNK_TOKEN

        self.reserved = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.index2word = self.reserved[:]
        self.word2index = dict(zip(self.reserved, range(len(self.reserved))))
        self.word2count = Counter()
        self.embeddings = None

    def build_vocab(self, vocab_counter, vocab_size=None, min_freq=1):
        self.word2count = vocab_counter
        self._add_words(vocab_counter.keys())
        self._trim(vocab_size=vocab_size, min_freq=min_freq)

    def _add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        assert len(self.word2index) == len(self.index2word)

    def _trim(self, vocab_size: int=None, min_freq: int=1):
        if min_freq <= 1 and (vocab_size is None or vocab_size >= len(self.word2index)):
            return
        ordered_words = sorted(((c, w) for (w, c) in self.word2count.items()), reverse=True)
        if vocab_size:
            ordered_words = ordered_words[:vocab_size]
        self.index2word = self.reserved[:]
        self.word2index = dict(zip(self.reserved, range(len(self.reserved))))
        self.word2count = Counter()
        for count, word in ordered_words:
            if count < min_freq: break
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.word2count[word] = count
                self.index2word.append(word)
        assert len(self.word2index) == len(self.index2word)

    def load_embeddings(self, file_path, scale=0.08, dtype=np.float32):
        hit_words = set()
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for line in f:
                line = line.split()
                word = line[0].decode('utf-8')
                idx = self.word2index.get(word.lower(), None)
                if idx is None or idx in hit_words:
                    continue

                vec = np.array(line[1:], dtype=dtype)
                if self.embeddings is None:
                    n_dims = len(vec)
                    self.embeddings = np.array(np.random.uniform(low=-scale, high=scale, size=(vocab_size, n_dims)), dtype=dtype)
                    self.embeddings[self.PAD] = np.zeros(n_dims)
                self.embeddings[idx] = vec
                hit_words.add(idx)
        print('Pretrained word embeddings hit ratio: {}'.format(len(hit_words) / len(self.index2word)))

    def randomize_embeddings(self, n_dims, scale=0.08):
        vocab_size = self.get_vocab_size()
        shape = (vocab_size, n_dims)
        self.embeddings = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)
        self.embeddings[self.PAD] = np.zeros(n_dims)

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    @lru_cache(maxsize=None)
    def is_word(self, token_id: int) -> bool:
        """Return whether the token at `token_id` is a word; False for punctuations."""
        if token_id < 4: return False
        if token_id >= len(self): return True  # OOV is assumed to be words
        token_str = self.index2word[token_id]
        if not word_detector.search(token_str) or token_str == '<P>':
            return False
        return True

    def get_vocab_size(self):
        return len(self.index2word)

    def getIndex(self, word):
        return self.word2index.get(word, self.UNK)

    def getWord(self, idx):
        return self.index2word[idx] if idx < len(self.index2word) else self.unk_token

    def to_word_sequence(self, seq):
        sentence = []
        for idx in seq:
            word = self.getWord(idx)
            sentence.append(word)
        return sentence

    def to_index_sequence(self, sentence):
        sentence = sentence.strip()
        seq = []
        for word in tokenize(sentence):
            idx = self.getIndex(word)
            seq.append(idx)
        return seq

    def to_index_sequence_for_list(self, words):
        seq = []
        for word in words:
            idx = self.getIndex(word)
            seq.append(idx)
        return seq

def collect_vocabs(all_instances):
    all_words = Counter()
    all_edge_types = Counter()
    for (sent1, sent2) in all_instances:
        # for each in sent1.words:
        #     all_words.update(each)
        for each in sent1.graph['g_features']:
            all_words.update(each)
        all_words.update(sent2.words)

        # for node, value in sent1.graph['g_adj'].items():
            # all_edge_types.update([each['edge'] for each in value])
    return all_words, all_edge_types
