from __future__ import unicode_literals
from io import StringIO
from os.path import join

import uuid
import os
import nltk
import pytest
import json
import numpy as np
import shutil
import itertools
import scipy.stats
import spacy
import pickle

from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer, StreamMethods
from spodernet.preprocessing.processors import Tokenizer, CustomTokenizer, SaveStateToList, AddToVocab, ToLower, ConvertTokenToIdx, SentTokenizer
from spodernet.preprocessing.processors import JsonLoaderProcessors, RemoveLineOnJsonValueCondition, DictKey2ListMapper
from spodernet.preprocessing.processors import StreamToHDF5, DeepSeqMap, StreamToBatch, TargetIdx2MultiTarget
from spodernet.preprocessing.processors import NERTokenizer, POSTokenizer, DependencyParser, TfidfFitter, TfidfTransformer
from spodernet.preprocessing.vocab import Vocab
from spodernet.preprocessing.batching import StreamBatcher, BatcherState
from spodernet.utils.util import get_data_path, load_data
from spodernet.utils.global_config import Config, Backends
from spodernet.hooks import LossHook, AccuracyHook, ETAHook

from spodernet.utils.logger import Logger, LogLevel
log = Logger('test_pipeline.py.txt')

Logger.GLOBAL_LOG_LEVEL = LogLevel.STATISTICAL
Logger.LOG_PROPABILITY = 0.1
Config.backend = Backends.TEST

def get_test_data_path_dict():
    paths = {}
    paths['snli10'] = './tests/test_data/snli10.json'
    paths['snli'] = './tests/test_data/snli.json'
    paths['snli3k'] = './tests/test_data/snli_3k.json'
    paths['snli1k'] = './tests/test_data/snli_1k.json'
    paths['wiki'] = './tests/test_data/wiki.json'

    return paths

def test_dict2listmapper():
    with open(join(get_data_path(), 'test.txt'), 'w') as f:
        for i in range(10):
            test_dict = {}
            test_dict['key1'] = str(i+5)
            test_dict['key2'] = str(i+3)
            test_dict['key3'] = str(i+4)
            f.write(json.dumps(test_dict) + '\n')

    s = DatasetStreamer()
    s.set_paths([join(get_data_path(), 'test.txt')])
    s.add_stream_processor(JsonLoaderProcessors())
    s.add_stream_processor(DictKey2ListMapper(['key3', 'key1', 'key2']))

    p = Pipeline('abc')
    p.add_text_processor(SaveStateToList('lines'))
    state = p.execute(s)
    for i, line in enumerate(state['data']['lines']['input']):
        assert int(line) == i+4, 'Input values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['support']):
        assert int(line) == i+5, 'Support values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['target']):
        assert int(line) == i+3, 'Target values does not correspond to the json key mapping.'

    os.remove(join(get_data_path(), 'test.txt'))
    shutil.rmtree(join(get_data_path(), 'abc'))

def test_remove_on_json_condition():
    with open(join(get_data_path(), 'test.txt'), 'w') as f:
        for i in range(10):
            test_dict = {}
            test_dict['key1'] = str(i+5)
            test_dict['key2'] = str(i+3)
            test_dict['key3'] = str(i+4)
            f.write(json.dumps(test_dict) + '\n')
            test_dict = {}
            test_dict['key1'] = str(i+5)
            test_dict['key2'] = str(i+3)
            test_dict['key3'] = 'remove me'
            f.write(json.dumps(test_dict) + '\n')

    s = DatasetStreamer()
    s.set_paths([join(get_data_path(), 'test.txt')])
    s.add_stream_processor(JsonLoaderProcessors())
    s.add_stream_processor(RemoveLineOnJsonValueCondition('key3', lambda inp: inp == 'remove me'))
    s.add_stream_processor(DictKey2ListMapper(['key3', 'key1', 'key2']))

    p = Pipeline('abc')
    p.add_text_processor(SaveStateToList('lines'))
    state = p.execute(s)

    assert len(state['data']['lines']['input']) == 10, 'Length different from filtered length!'
    for i, line in enumerate(state['data']['lines']['input']):
        assert int(line) == i+4, 'Input values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['support']):
        assert int(line) == i+5, 'Support values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['target']):
        assert int(line) == i+3, 'Target values does not correspond to the json key mapping.'

    os.remove(join(get_data_path(), 'test.txt'))
    shutil.rmtree(join(get_data_path(), 'abc'))


def test_tokenization():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_sent_processor(SaveStateToList('tokens'))
    state = p.execute(s)

    inp_sents = state['data']['tokens']['input']
    sup_sents = state['data']['tokens']['support']
    sents = inp_sents + sup_sents
    log.statistical('input sentence of tokens: {0}', 0.5, inp_sents[0])
    log.statistical('support sentence of tokens: {0}', 0.5, sup_sents[0])

    # 2. setup nltk tokenization
    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : []}
        for line in f:
            inp, sup, t = json.loads(line)
            tokenized_sents['input'].append(tokenizer.tokenize(inp))
            tokenized_sents['support'].append(tokenizer.tokenize(sup))
            log.statistical('input sentence of tokens: {0}', 0.01, tokenized_sents['input'][-1])
            log.statistical('support sentence of tokens: {0}', 0.01, tokenized_sents['support'][-1])

    sents_nltk = tokenized_sents['input'] + tokenized_sents['support']
    # 3. test equality
    assert len(sents) == len(sents_nltk), 'Sentence count differs!'
    log.debug('count should be 200: {0}', len(sents))
    for sent1, sent2 in zip(sents, sents_nltk):
        assert len(sent1) == len(sent2), 'Token count differs!'
        log.statistical('a sentence of tokens: {0}', 0.01, sent1)
        for token1, token2 in zip(sent1, sent2):
            assert token1 == token2, 'Token values differ!'
            log.statistical('a token: {0}', 0.001, token1)


def test_sent_tokenizer():
    path = get_test_data_path_dict()['wiki']
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    s = DatasetStreamer()
    s.set_path(path)
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_text_processor(SentTokenizer())
    p.add_post_processor(SaveStateToList('docs'))
    state = p.execute(s)

    # 2. setup manual sentence -> token processing
    docs = state['data']['docs']['support']

    docs_nltk = []
    with open(path) as f:
        for line in f:
            sup_sents = []
            inp, sup, t = json.loads(line)
            for sent in sent_tokenizer.tokenize(sup):
                sup_sents.append(sent)
            docs_nltk.append(sup_sents)


    # 3. test equivalence
    assert len(docs) == len(docs_nltk) == 3, 'Differernt amount of documents'
    for sents1, sents2 in zip(docs, docs_nltk):
        assert len(sents1) == len(sents2), 'Sentence count differs!'
        for sent1, sent2, in zip(sents1, sents2):
            # index for token level which was skipped
            assert sent1[0] == sent2, 'Sentence differs!'


def test_nltk_tokenizer():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_sent_processor(SaveStateToList('tokens'))
    state = p.execute(s)

    inp_sents = state['data']['tokens']['input']
    sup_sents = state['data']['tokens']['support']
    sents = inp_sents + sup_sents

    p.clear_processors()
    p.add_sent_processor(Tokenizer())
    p.add_sent_processor(SaveStateToList('tokens_nltk'))
    state = p.execute(s)

    inp_sents = state['data']['tokens_nltk']['input']
    sup_sents = state['data']['tokens_nltk']['support']
    sents_nltk = inp_sents + sup_sents

    # 3. test equality
    misalignment_count = 0
    assert len(sents) == len(sents_nltk), 'Sentence count differs!'
    log.debug('count should be 200: {0}', len(sents))
    for sent1, sent2 in zip(sents, sents_nltk):
        assert len(sent1)*0.8 < len(sent2) and len(sent1)*1.2 > len(sent2), 'Token count differs substantially!'
        for token1, token2 in zip(sent1, sent2):
            if token1 != token2:
                misalignment_count +=1
                break
    assert misalignment_count < 20, 'To many misalignments between tokenizers!'

def test_path_creation():
    names = []
    for i in range(100):
        names.append(str(uuid.uuid4()))

    for name in names:
        p = Pipeline(name)

    home = os.environ['HOME']
    paths = [ join(home, '.data', name) for name in names]
    for path in paths:
        assert os.path.exists(path)
        os.rmdir(path)

def test_vocab():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    state = p.execute(s)

    # 1. use Vocab manually and test it against manual vocabulary
    idx2token = {}
    token2idx = {}
    token2idx['OOV'] = 0
    idx2token[0] = 'OOV'
    # empty = 0
    token2idx[''] = 1
    idx2token[1] = ''
    idx = 2
    v = Vocab('test')
    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : []}
        for line in f:
            inp, sup, t = json.loads(line)

            for token in tokenizer.tokenize(inp):
                v.add_token(token)
                if token not in token2idx:
                    token2idx[token] = idx
                    idx2token[idx] = token
                    idx += 1
                    log.statistical('uncommon word if high number: {0}, {1}', 0.001, token, idx)
                    log.statistical('uncommon word if high number: {0}, {1}', 0.001, token, v.get_idx(token))

            for token in tokenizer.tokenize(sup):
                v.add_token(token)
                if token not in token2idx:
                    token2idx[token] = idx
                    idx2token[idx] = token
                    idx += 1
                    log.statistical('uncommon word if high number: {0}, {1}', 0.001, token, idx)
                    log.statistical('uncommon word if high number: {0}, {1}', 0.001, token, v.get_idx(token))

            v.add_label(t)
            log.statistical('label vocab index, that is small numbers: {0}', 0.01, v.idx2label.keys())


    # 3. Compare vocabs
    v2 = state['vocab']['general']
    for token in v.token2idx:
        assert v.token2idx[token] == v2.token2idx[token], 'Index for token not the same!'
        assert v.token2idx[token] == token2idx[token], 'Index for token not the same!'

    for idx in v.idx2token:
        assert v.idx2token[idx] == v2.idx2token[idx], 'Token for index not the same!'
        assert v.idx2token[idx] == idx2token[idx], 'Token for index not the same!'

    for label in v.label2idx:
        log.statistical('a label: {0}', 0.001, label)
        assert v.label2idx[label] == v2.label2idx[label], 'Index for label not the same!'

    for idx in v.idx2label:
        assert v.idx2label[idx] == v2.idx2label[idx], 'Label for index not the same!'


def test_separate_vocabs():

    # 1. write test data
    file_path = join(get_data_path(), 'test_pipeline', 'test_data.json')
    with open(file_path, 'w') as f:
        f.write(json.dumps(['0', 'a','-']) + '\n')
        f.write(json.dumps(['1', 'b','&']) + '\n')
        f.write(json.dumps(['2', 'c','#']) + '\n')

    s = DatasetStreamer()
    s.set_path(file_path)
    s.add_stream_processor(JsonLoaderProcessors())
    # 2. read test data with pipeline
    p = Pipeline('test_pipeline')

    p.add_token_processor(AddToVocab())
    state = p.execute(s)
    vocab = state['vocab']['general']
    inp_vocab = state['vocab']['input']
    sup_vocab = state['vocab']['support']
    tar_vocab = state['vocab']['target']

    # 6 token + empty and unknown = 8 
    assert vocab.num_token == 6 + 2, 'General vocab token count should be 8, but was {0} instead.'.format(vocab.num_token)
    assert vocab.num_labels == 3, 'General vocab token count should be 3, but was {0} instead.'.format(vocab.num_labels)

    assert inp_vocab.num_token == 3 + 2, 'General vocab token count should be 5, but was {0} instead.'.format(inp_vocab.num_token)
    assert inp_vocab.num_labels == 0, 'General vocab token count should be 0, but was {0} instead.'.format(inp_vocab.num_labels)
    assert sup_vocab.num_token == 3 + 2, 'General vocab token count should be 5, but was {0} instead.'.format(sup_vocab.num_token)
    assert sup_vocab.num_labels == 0, 'General vocab token count should be 0, but was {0} instead.'.format(sup_vocab.num_labels)
    assert tar_vocab.num_token == 3 + 2, 'General vocab token count should be 5, but was {0} instead.'.format(tar_vocab.num_token)
    assert tar_vocab.num_labels == 0, 'General vocab token count should be 0, but was {0} instead.'.format(tar_vocab.num_labels)

    for token in ['0', '1', '2']:
        assert token in vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)
        assert token in inp_vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)

    for token in ['a', 'b', 'c']:
        assert token in vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)
        assert token in sup_vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)

    for token in ['-', '&', '#']:
        assert token in vocab.label2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)
        assert token in tar_vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)


def test_to_lower_sent():
    path = get_test_data_path_dict()['snli']

    s = DatasetStreamer()
    s.set_path(path)
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_sent_processor(ToLower())
    p.add_sent_processor(SaveStateToList('sents'))
    state = p.execute(s)

    inp_sents = state['data']['sents']['input']
    sup_sents = state['data']['sents']['support']
    sents = inp_sents + sup_sents

    # 2. test lowercase
    assert len(sents) == 200 # we have 100 samples for snli
    for sent in sents:
        log.statistical('lower case sentence {0}', 0.001, sent)
        assert sent == sent.lower(), 'Sentence is not lower case'

def test_to_lower_token():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    path = get_test_data_path_dict()['snli']

    s = DatasetStreamer()
    s.set_path(path)
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_token_processor(ToLower())
    p.add_token_processor(SaveStateToList('tokens'))
    state = p.execute(s)

    inp_tokens = state['data']['tokens']['input']
    sup_tokens = state['data']['tokens']['support']
    tokens = inp_tokens + sup_tokens

    # 2. test lowercase
    for token in tokens:
        log.statistical('lower case token: {0}', 0.0001, token)
        assert token == token.lower(), 'Token is not lower case'

def test_save_to_list_text():
    path = get_test_data_path_dict()['wiki']

    s = DatasetStreamer()
    s.set_path(path)
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_text_processor(SaveStateToList('text'))
    state = p.execute(s)

    inp_texts = state['data']['text']['input']
    sup_texts = state['data']['text']['support']
    assert len(inp_texts) == 3, 'The input data size should be three samples, but found {0}'.format(len(inp_texts))
    assert len(inp_texts) == 3, 'The input data size should be three samples, but found {0}'.format(len(sup_texts))
    with open(path) as f:
        for inp1, sup1, line in zip(inp_texts, sup_texts, f):
            inp2, sup2, t = json.loads(line)
            log.statistical('a wikipedia paragraph: {0}', 0.5, sup1)
            assert inp1 == inp2, 'Saved text data not the same!'
            assert sup1 == sup2, 'Saved text data not the same!'


def test_save_to_list_sentences():
    path = get_test_data_path_dict()['wiki']
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    s = DatasetStreamer()
    s.set_path(path)
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_text_processor(CustomTokenizer(sent_tokenizer.tokenize))
    p.add_sent_processor(SaveStateToList('sents'))
    state = p.execute(s)

    # 2. setup manual sentence processing
    inp_sents = state['data']['sents']['input']
    sup_sents = state['data']['sents']['support']
    inp_sents2 = []
    sup_sents2 = []
    with open(path) as f:
        for line in f:
            inp, sup, t = json.loads(line)
            sup_sents2 += sent_tokenizer.tokenize(sup)
            inp_sents2 += sent_tokenizer.tokenize(inp)
            log.statistical('a list of sentences: {0}', 0.3, sup_sents)

    # 3. test equivalence
    assert len(inp_sents) == len(inp_sents2), 'Sentence count differs!'
    assert len(sup_sents) == len(sup_sents2), 'Sentence count differs!'

    for sent1, sent2 in zip(inp_sents, inp_sents2):
        assert sent1 == sent2, 'Saved sentence data not the same!'

    for sent1, sent2 in zip(sup_sents, sup_sents2):
        log.statistical('a sentence from a wiki paragraph: {0}', 0.3, sent1)
        assert sent1 == sent2, 'Saved sentence data not the same!'


def test_save_to_list_post_process():
    path = get_test_data_path_dict()['wiki']
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    s = DatasetStreamer()
    s.set_path(path)
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_text_processor(CustomTokenizer(sent_tokenizer.tokenize))
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveStateToList('samples'))
    state = p.execute(s)

    # 2. setup manual sentence -> token processing
    inp_samples = state['data']['samples']['input']
    sup_samples = state['data']['samples']['support']
    inp_samples2 = []
    sup_samples2 = []
    with open(path) as f:
        for line in f:
            sup_sents = []
            inp_sents = []
            inp, sup, t = json.loads(line)
            for sent in sent_tokenizer.tokenize(sup):
                sup_sents.append(tokenizer.tokenize(sent))
            for sent in sent_tokenizer.tokenize(inp):
                inp_sents.append(tokenizer.tokenize(sent))
            inp_samples2.append(inp_sents)
            sup_samples2.append(sup_sents)


    # 3. test equivalence
    for sample1, sample2 in zip(inp_samples, inp_samples2):
        assert len(sample1) == len(sample2), 'Sentence count differs!'
        for sent1, sent2, in zip(sample1, sample2):
            assert len(sent1) == len(sent2), 'Token count differs!'
            for token1, token2 in zip(sent1, sent2):
                assert token1 == token2, 'Tokens differ!'

    for sample1, sample2 in zip(sup_samples, sup_samples2):
        log.statistical('a wiki paragraph {0}', 0.1,  sample1)
        assert len(sample1) == len(sample2), 'Sentence count differs!'
        for sent1, sent2, in zip(sample1, sample2):
            log.statistical('a sentence of tokens of a wiki paragraph {0}', 0.01, sent1)
            assert len(sent1) == len(sent2), 'Token count differs!'
            for token1, token2 in zip(sent1, sent2):
                log.statistical('a token from a sentence of a wiki paragraph {0}', 0.001, token1)
                assert token1 == token2, 'Tokens differ!'



def test_convert_token_to_idx_no_sentences():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    state = p.execute(s)

    inp_indices = state['data']['idx']['input']
    label_idx = state['data']['idx']['target']
    log.statistical('a list of about 10 indices: {0}', 0.5, inp_indices[0])

    # 2. use Vocab manually
    v = Vocab('test')
    with open(get_test_data_path_dict()['snli']) as f:
        for line in f:
            inp, sup, t = json.loads(line)

            for token in tokenizer.tokenize(inp):
                v.add_token(token)

            for token in tokenizer.tokenize(sup):
                v.add_token(token)

            v.add_label(t)

    # 3. index manually
    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : [], 'target' : []}
        for line in f:
            inp_idx = []
            sup_idx = []

            inp, sup, t = json.loads(line)

            for token in tokenizer.tokenize(inp):
                inp_idx.append(v.get_idx(token))

            for token in tokenizer.tokenize(sup):
                sup_idx.append(v.get_idx(token))

            log.statistical('a list of about 10 indices {0}', 0.01, inp_idx)
            tokenized_sents['target'].append(v.get_idx_label(t))
            tokenized_sents['input'].append(inp_idx)
            tokenized_sents['support'].append(sup_idx)


    # 4. Compare idx
    assert len(tokenized_sents['input']) == len(inp_indices), 'Sentence count differs!'
    for sent1, sample in zip(tokenized_sents['input'], inp_indices):
        sent2 = sample[0] # in this case we do not have sentences
        assert len(sent1) == len(sent2), 'Index count (token count) differs!'
        for idx1, idx2 in zip(sent1, sent2):
            assert idx1 == idx2, 'Index for token differs!'

    # 5. Compare label idx
    for idx1, sample in zip(tokenized_sents['target'], label_idx):
        # sample[0] == sent
        # sent[0] = idx
        assert idx1 == sample[0][0], 'Index for label differs!'


def test_convert_to_idx_with_separate_vocabs():

    # 1. write test data
    file_path = join(get_data_path(), 'test_pipeline', 'test_data.json')
    with open(file_path, 'w') as f:
        f.write(json.dumps(['0', 'a','-']) + '\n')
        f.write(json.dumps(['1', 'b','&']) + '\n')
        f.write(json.dumps(['2', 'c','#']) + '\n')

    # 2. read test data with pipeline
    keys2keys = {}
    keys2keys['input'] = 'input'
    keys2keys['support'] = 'support'

    s = DatasetStreamer()
    s.set_path(file_path)
    s.add_stream_processor(JsonLoaderProcessors())

    p = Pipeline('test_pipeline')
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys))
    p.add_post_processor(SaveStateToList('idx'))
    state = p.execute(s)

    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['input']

    # 0 = UNK, 1 = '', 2,3,4 -> max index is 4
    assert np.max(inp_indices) == 2 + 2, 'Max idx should have been 2 if the vocabularies were separates!'
    assert np.max(sup_indices) == 2 + 2, 'Max idx should have been 2 if the vocabularies were separates!'

def test_save_lengths():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    state = p.execute(s)

    lengths_inp = state['data']['lengths']['input']
    lengths_sup = state['data']['lengths']['support']
    log.statistical('a list of length values {0}', 0.5, lengths_inp)
    lengths1 = lengths_inp + lengths_sup

    # 2. generate lengths manually
    lengths_inp2 = []
    lengths_sup2 = []
    with open(get_test_data_path_dict()['snli']) as f:
        for line in f:
            inp, sup, t = json.loads(line)

            lengths_inp2.append(len(tokenizer.tokenize(inp)))
            lengths_sup2.append(len(tokenizer.tokenize(sup)))

    lengths2 = lengths_inp2 + lengths_sup2

    # 3. test for equal lengths
    assert len(lengths1) == len(lengths2), 'Count of lengths differs!'
    assert len(lengths1) == 200, 'Count of lengths not as expected for SNLI test data!'
    for l1, l2 in zip(lengths1, lengths2):
        assert l1 == l2, 'Lengths of sentence differs!'


def test_stream_to_hdf5():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'snli_test'
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_token_processor(AddToVocab())
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    streamer = StreamToHDF5(data_folder_name, samples_per_file=2, keys=['input', 'support', 'target'])
    p.add_post_processor(streamer)
    state = p.execute(s)

    # 2. Load data from the SaveStateToList hook
    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['support']
    t_indices = state['data']['idx']['target']
    max_inp_len = np.max(state['data']['lengths']['input'])
    max_sup_len = np.max(state['data']['lengths']['support'])
    # For SNLI the targets consist of single words'
    assert np.max(state['data']['lengths']['target']) == 1, 'Max index label length should be 1'
    assert 'counts' in streamer.config, 'counts key not found in config dict!'
    assert len(streamer.config['counts']) > 0,'Counts of samples per file must be larger than zero (probably no files have been saved)'

    # 3. parse data to numpy
    n = len(inp_indices)
    X = np.zeros((n, max_inp_len), dtype=np.int64)
    X_len = np.zeros((n), dtype=np.int64)
    S = np.zeros((n, max_sup_len), dtype=np.int64)
    S_len = np.zeros((n), dtype=np.int64)
    t = np.zeros((n, 1), dtype=np.int64)
    index = np.zeros((n), dtype=np.int64)

    for i in range(len(inp_indices)):
        sample_inp = inp_indices[i][0]
        sample_sup = sup_indices[i][0]
        sample_t = t_indices[i][0]
        l = len(sample_inp)
        X_len[i] = l
        X[i, :l] = sample_inp

        l = len(sample_sup)
        S_len[i] = l
        S[i, :l] = sample_sup

        t[i] = sample_t[0]
        index[i] = i

    # 4. setup expected paths
    inp_paths = [join(base_path, 'input_' + str(i) + '.hdf5') for i in range(1, 50)]
    sup_paths = [join(base_path, 'support_' + str(i) + '.hdf5') for i in range(1, 50)]
    target_paths = [join(base_path, 'target_' + str(i) + '.hdf5') for i in range(1, 50)]
    inp_len_paths = [join(base_path, 'input_lengths_' + str(i) + '.hdf5') for i in range(1, 50)]
    sup_len_paths = [join(base_path, 'support_lengths_' + str(i) + '.hdf5') for i in range(1, 50)]
    index_paths = [join(base_path, 'index_' + str(i) + '.hdf5') for i in range(1, 50)]

    data_idx = 0
    for path in index_paths:
        assert os.path.exists(path), 'Index path does not exist!'
        start = data_idx*2
        end = (data_idx + 1)*2
        data_idx += 1
        index[start:end] = load_data(path)

    X = X[index]
    S = S[index]
    t = t[index]
    X_len = X_len[index]
    S_len = S_len[index]
    zip_iter = zip([X, S, t, X_len, S_len], [inp_paths, sup_paths, target_paths, inp_len_paths, sup_len_paths ])

    # 5. Compare data
    for data, paths in zip_iter:
        data_idx = 0
        for path in paths:
            folder, filename = os.path.split(path)
            assert (os.path.exists(path) or os.path.exists(join(folder, 'indptr_' + filename))), 'This file should have been created by the HDF5Streamer: {0}'.format(path)
            shard = load_data(path)
            start = data_idx*2
            end = (data_idx + 1)*2
            np.testing.assert_array_equal(shard, data[start:end], 'HDF5 Stream data not equal for path {0}'.format(path))
            data_idx += 1

    # 6. compare config
    config_path = join(base_path, 'hdf5_config.pkl')
    config_reference = streamer.config
    assert os.path.exists(config_path), 'No HDF5 config exists under the path: {0}'.format(config_path)
    config_dict = pickle.load(open(config_path, 'rb'))
    assert 'paths' in config_dict, 'paths key not found in config dict!'
    assert 'fractions' in config_dict, 'fractions key not found in config dict!'
    assert 'counts' in config_dict, 'counts key not found in config dict!'
    for paths1, paths2 in zip(config_dict['paths'], streamer.config['paths']):
        for path1, path2 in zip(paths1, paths2):
            assert path1 == path2, 'Paths differ from HDF5 config!'
    np.testing.assert_array_equal(config_dict['fractions'], streamer.config['fractions'], 'Fractions for HDF5 samples per file not equal!')
    np.testing.assert_array_equal(config_dict['counts'], streamer.config['counts'], 'Counts for HDF5 samples per file not equal!')
    assert len(streamer.config['counts']) > 0, 'List of counts empty!'

    path_types = ['input', 'support', 'input_length', 'support_length', 'target', 'target_length', 'index']
    for i, paths in enumerate(streamer.config['paths']):
        assert len(paths) == 7, 'One path type is missing! Required path types {0}, existing paths {1}.'.format(path_types, paths)

    # 7. clean up
    shutil.rmtree(base_path)

batch_size = [17, 128]
samples_per_file = [500]
randomize = [True, False]
test_data = [r for r in itertools.product(samples_per_file, randomize, batch_size)]
test_data.append((1000000, True, 83))
str_func = lambda i, j, k: 'samples_per_file={0}, randomize={1}, batch_size={2}'.format(i, j, k)
ids = [str_func(i,j,k) for i,j,k in test_data]
test_idx = np.random.randint(0,len(test_data),3)
@pytest.mark.parametrize("samples_per_file, randomize, batch_size", test_data, ids=ids)
def test_non_random_stream_batcher(samples_per_file, randomize, batch_size):
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'snli_test'
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli1k'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_token_processor(AddToVocab())
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    streamer = StreamToHDF5(data_folder_name, samples_per_file=samples_per_file, keys=['input', 'support', 'target'])
    p.add_post_processor(streamer)
    state = p.execute(s)

    # 2. Load data from the SaveStateToList hook
    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['support']
    t_indices = state['data']['idx']['target']
    max_inp_len = np.max(state['data']['lengths']['input'])
    max_sup_len = np.max(state['data']['lengths']['support'])
    # For SNLI the targets consist of single words'
    assert np.max(state['data']['lengths']['target']) == 1, 'Max index label length should be 1'
    assert 'counts' in streamer.config, 'counts key not found in config dict!'
    assert len(streamer.config['counts']) > 0,'Counts of samples per file must be larger than zero (probably no files have been saved)'
    if samples_per_file == 100000:
        count = len(streamer.config['counts'])
        assert count == 1,'Samples per files is 100000 and there should be one file for 1k samples, but there are {0}'.format(count)

    assert streamer.num_samples == 1000, 'There should be 1000 samples for this dataset, but found {1}!'.format(streamer.num_samples)


    # 3. parse data to numpy
    n = len(inp_indices)
    X = np.zeros((n, max_inp_len), dtype=np.int64)
    X_len = np.zeros((n), dtype=np.int64)
    S = np.zeros((n, max_sup_len), dtype=np.int64)
    S_len = np.zeros((n), dtype=np.int64)
    T = np.zeros((n, 1), dtype=np.int64)

    for i in range(len(inp_indices)):
        sample_inp = inp_indices[i][0]
        sample_sup = sup_indices[i][0]
        sample_t = t_indices[i][0]
        l = len(sample_inp)
        X_len[i] = l
        X[i, :l] = sample_inp

        l = len(sample_sup)
        S_len[i] = l
        S[i, :l] = sample_sup

        T[i] = sample_t[0]

    epochs = 2
    batcher = StreamBatcher(pipeline_folder, data_folder_name, batch_size, loader_threads=8, randomize=randomize)
    del batcher.at_batch_prepared_observers[:]

    # 4. test data equality
    for epoch in range(epochs):
        for i, (x, x_len, s, s_len, t, t_len, idx) in enumerate(batcher):
            assert np.int32 == x_len.dtype, 'Input length type should be int32!'
            assert np.int32 == s_len.dtype, 'Support length type should be int32!'
            assert np.int32 == x.dtype, 'Input type should be int32!'
            assert np.int32 == s.dtype, 'Input type should be int32!'
            assert np.int32 == t.dtype, 'Target type should be int32!'
            assert np.int32 == idx.dtype, 'Index type should be int32!'
            np.testing.assert_array_equal(X_len[idx], x_len, 'Input length data not equal!')
            np.testing.assert_array_equal(S_len[idx], s_len, 'Support length data not equal!')
            np.testing.assert_array_equal(X[idx], x, 'Input data not equal!')
            np.testing.assert_array_equal(S[idx], s, 'Support data not equal!')
            np.testing.assert_array_equal(T[idx], t, 'Target data not equal!')

    # 5. clean up
    shutil.rmtree(base_path)

def test_abitrary_input_data():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    base_path = join(get_data_path(), 'test_keys')
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    file_path = join(get_data_path(), 'test_keys', 'test_data.json')

    questions = [['bla bla Q1', 'this is q2', 'q3'], ['q4 set2', 'or is it q1?']]
    support = [['I like', 'multiple supports'], ['yep', 'they are pretty cool', 'yeah, right?']]
    answer = [['yes', 'absolutly', 'not really'], ['you bet', 'yes']]
    pos_tag = [['t1', 't2'], ['t1', 't2', 't3']]

    keys2keys = {}
    keys2keys['answer'] = 'answer'
    keys2keys['pos'] = 'pos'

    p = Pipeline('test_keys', keys=['question', 'support', 'answer', 'pos'])
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab(general_vocab_keys=['question', 'support']))

    with open(file_path, 'w') as f:
        for i in range(2):
            f.write(json.dumps([questions[i], support[i], answer[i], pos_tag[i]]) + '\n')

    s = DatasetStreamer(input_keys=['question', 'support', 'answer', 'pos'])
    s.set_path(file_path)
    s.add_stream_processor(JsonLoaderProcessors())

    p.execute(s)
    p.clear_processors()
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_token_processor(ConvertTokenToIdx(keys2keys=keys2keys))
    p.add_post_processor(StreamToHDF5('test', keys=['question', 'support', 'answer', 'pos']))
    p.add_post_processor(SaveStateToList('data'))
    state = p.execute(s)

    Q = state['data']['data']['question']
    S = state['data']['data']['support']
    A = state['data']['data']['answer']
    pos = state['data']['data']['pos']
    #vocab is offset by 2, due to UNK and empty word ''
    # note that we travers the data like q1, s1, a1; q2, s2, a2
    # we share vocab between question and support
    expected_Q_ids = [[[ 2, 2, 3], [4, 5, 6], [7]], [[12, 13], [14, 5, 15, 16, 17]]]
    expected_S_ids = [[[8, 9], [10, 11]], [[18], [19, 20, 21, 22], [23, 24, 25, 17]]]
    expected_answer_ids = [[[2],[3],[4, 5]],[[6,7], [2]]]
    expected_pos_ids = [[[2],[3]],[[2],[3],[4]]]

    np.testing.assert_array_equal(np.array(expected_Q_ids), Q)
    np.testing.assert_array_equal(np.array(expected_S_ids), S)
    np.testing.assert_array_equal(np.array(expected_answer_ids), A)
    np.testing.assert_array_equal(np.array(expected_pos_ids), pos)


@pytest.mark.skip(reason='outdated code; remove the bin streamer class?')
def test_bin_streamer():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'bin_snli_test'
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
    batch_size = 4
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli1k'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.execute(s)
    p.clear_processors()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    bin_creator = CreateBinsByNestedLength(data_folder_name, min_batch_size=batch_size, raise_on_throw_away_fraction=0.5)
    p.add_post_processor(bin_creator)
    state = p.execute(s)

    # 2. Load data from the SaveStateToList hook
    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['support']
    t_indices = state['data']['idx']['target']
    max_inp_len = np.max(state['data']['lengths']['input'])
    max_sup_len = np.max(state['data']['lengths']['support'])
    # For SNLI the targets consist of single words'
    assert np.max(state['data']['lengths']['target']) == 1, 'Max index label length should be 1'

    # 3. parse data to numpy
    n = len(inp_indices)
    X = np.zeros((n, max_inp_len), dtype=np.int64)
    X_len = np.zeros((n), dtype=np.int64)
    S = np.zeros((n, max_sup_len), dtype=np.int64)
    S_len = np.zeros((n), dtype=np.int64)
    T = np.zeros((n), dtype=np.int64)

    for i in range(len(inp_indices)):
        sample_inp = inp_indices[i][0]
        sample_sup = sup_indices[i][0]
        sample_t = t_indices[i][0]
        l = len(sample_inp)
        X_len[i] = l
        X[i, :l] = sample_inp

        l = len(sample_sup)
        S_len[i] = l
        S[i, :l] = sample_sup

        T[i] = sample_t[0]

    epochs = 3
    batcher = StreamBatcher(pipeline_folder, data_folder_name, batch_size, loader_threads=8, randomize=True)
    del batcher.at_batch_prepared_observers[:] # we want to test on raw numpy data

    # 4. test data equality
    for epoch in range(epochs):
        for x, x_len, s, s_len, t, idx in batcher:
            assert np.int32 == x_len.dtype, 'Input length type should be int32!'
            assert np.int32 == s_len.dtype, 'Support length type should be int32!'
            assert np.int32 == x.dtype, 'Input type should be int32!'
            assert np.int32 == s.dtype, 'Input type should be int32!'
            assert np.int32 == t.dtype, 'Target type should be int32!'
            assert np.int32 == idx.dtype, 'Index type should be int32!'
            np.testing.assert_array_equal(X[idx, :x_len[0]], x, 'Input data not equal!')
            np.testing.assert_array_equal(S[idx, :s_len[0]], s, 'Support data not equal!')
            np.testing.assert_array_equal(X_len[idx], x_len, 'Input length data not equal!')
            np.testing.assert_array_equal(S_len[idx], s_len, 'Support length data not equal!')
            np.testing.assert_array_equal(T[idx], t, 'Target data not equal!')

            # if the next tests fail, it means the batches provides the wrong length for the sample
            np.testing.assert_array_equal(S[idx, s_len[0]:], np.zeros((batch_size, S.shape[1]-s_len[0])), 'Support tail not padded exclusively with zeros!')
            np.testing.assert_array_equal(X[idx, x_len[0]:], np.zeros((batch_size, X.shape[1]-x_len[0])), 'Input tail not padded exclusively with zeros!')

    # 5. clean up
    shutil.rmtree(base_path)


names = ['loss', 'accuracy']
print_every = [20, 7, 13, 2000]
test_data = [r for r in itertools.product(names, print_every)]
ids = ['name={0}, print_every={1}'.format(name, print_every) for name, print_every in test_data]
@pytest.mark.parametrize("hook_name, print_every", test_data, ids=ids)
def test_hook(hook_name, print_every):
    def calc_confidence_interval(expected_loss):
        assert len(expected_loss) > 0
        mean = np.mean(expected_loss)
        print(expected_loss, mean)
        std = np.std(expected_loss)
        z = scipy.stats.norm.ppf(0.99)
        se = z*std/np.sqrt(print_every)
        lower_expected = mean-se
        upper_expected = mean+se
        return lower_expected, upper_expected, mean, n

    def generate_loss():
        loss = np.random.rand()
        state = BatcherState()
        state.loss = loss
        return loss, state

    def generate_accuracy():
        target = np.random.randint(0,3, print_every)
        argmax = np.random.randint(0,3, print_every)
        state = BatcherState()
        state.targets = target
        state.argmax = argmax
        mean = np.mean(expected_loss)
        accuracy = np.mean(target==argmax)
        return accuracy, state

    if hook_name == 'loss':
        hook = LossHook(print_every_x_batches=print_every)
        gen_func = generate_loss
    elif hook_name == 'accuracy':
        hook = AccuracyHook(print_every_x_batches=print_every)
        gen_func = generate_accuracy

    expected_loss = []
    state = BatcherState()
    for epoch in range(2):
        for i in range(113):
            metric, state = gen_func()
            expected_loss.append(metric)
            lower, upper, m, n = hook.at_end_of_iter_event(state)
            if (i+1) % print_every == 0:
                lower_expected, upper_expected, mean, n2 = calc_confidence_interval(expected_loss)
                assert n == n2, 'Sample size not equal!'
                assert np.allclose(m, mean), 'Mean not equal!'
                assert np.allclose(lower, lower_expected), 'Lower confidence bound not equal!'
                assert np.allclose(upper, upper_expected), 'Upper confidence bound not equal!'
                del expected_loss[:]

        lower, upper, m, n = hook.at_end_of_epoch_event(state)
        lower_expected, upper_expected, mean, n2 = calc_confidence_interval(expected_loss)
        del expected_loss[:]


def test_variable_duplication():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder)
    batch_size = 32
    func = lambda x: [word[0:2] for word in x]
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    s = DatasetStreamer(input_keys=['input', 'support', 'target'], output_keys=['input', 'support', 'target', 'input'])
    s.set_path(get_test_data_path_dict()['snli'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. Setup pipeline to save lengths and generate vocabulary
    keys = ['input', 'support', 'target', 'input_pos']
    p = Pipeline(pipeline_folder, keys=keys)
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_sent_processor(SaveStateToList('tokens'))
    p.add_post_processor(DeepSeqMap(func), keys=['input_pos'])
    p.add_post_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx(keys2keys={'input_pos' : 'input_pos'}))
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    state = p.execute(s)

    # 2. Load data from the SaveStateToList hook
    inp_sents = state['data']['tokens']['input']
    pos_tags = state['data']['idx']['input_pos']
    vocab = p.state['vocab']['input_pos']


    tags_expected = []
    for sent in inp_sents:
        tag = [word[:2] for word in sent]
        tags_expected.append(tag)

    tags = []
    for sent in pos_tags[0]:
        tag = [vocab.get_word(idx) for idx in sent]
        tags.append(tag)


    for tags1, tags2 in zip(tags, tags_expected):
        assert len(tags1) == len(tags2), 'POS tag lengths not the same!'
        for tag1, tag2 in zip(tags1, tags2):
            assert tag1 == tag2, 'POS tags were not the same!'

    # 5. clean up
    shutil.rmtree(base_path)

def test_stream_to_batch():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    path = get_test_data_path_dict()['snli']
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder)
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    s = DatasetStreamer()
    s.set_path(path)
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline(pipeline_folder)
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_token_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('samples'))
    state = p.execute(s)
    p.save_vocabs()

    # testing if we can pass data through a "pre-trained" pipeline and get the right results
    p2 = Pipeline(pipeline_folder)
    p2.load_vocabs()
    p2.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p2.add_token_processor(ConvertTokenToIdx())

    batcher = StreamToBatch()
    p2.add_post_processor(batcher)
    p2.execute(s)


    # 2. setup manual sentence -> token processing
    inp_samples = state['data']['samples']['input']
    sup_samples = state['data']['samples']['support']
    str2var = batcher.get_data()

    for x1, x2, x2len in zip(inp_samples, str2var['input'], str2var['input_length']):
        np.testing.assert_array_equal(x1[0], x2[:x2len], 'Array length not equal')
        assert np.sum(x2[x2len:]) == 0, 'Not padded with zeros!'
        assert x2len == len(x1[0]), 'Lengths not equal!'

    shutil.rmtree(base_path)


def test_stream_from_data():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    path = get_test_data_path_dict()['snli']
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder)
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    s = DatasetStreamer()
    s.set_path(path)
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline(pipeline_folder)
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_token_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('samples'))
    state = p.execute(s)
    p.save_vocabs()

    # testing if we can pass data through a "pre-trained" pipeline and get the right results
    p2 = Pipeline(pipeline_folder)
    p2.load_vocabs()
    p2.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p2.add_token_processor(ConvertTokenToIdx())

    batcher = StreamToBatch()
    p2.add_post_processor(batcher)
    p2.execute(s)

    s2 = DatasetStreamer(stream_method=StreamMethods.data)

    with open(path) as f:
        data = f.readlines()

    s2.set_data(data)
    s2.add_stream_processor(JsonLoaderProcessors())
    p3 = Pipeline(pipeline_folder)
    p3.load_vocabs()
    p3.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p3.add_token_processor(ConvertTokenToIdx())

    batcher2 = StreamToBatch()
    p3.add_post_processor(batcher2)
    p3.execute(s2)


    # 2. setup manual sentence -> token processing
    inp_samples = state['data']['samples']['input']
    sup_samples = state['data']['samples']['support']
    str2var = batcher.get_data()
    str2var2= batcher2.get_data()

    for key in str2var.keys():
        assert key in str2var2, 'Data batching method does not result in the same keys!'

    for key in str2var.keys():
        var1 = str2var[key]
        var2 = str2var2[key]
        np.testing.assert_array_equal(var1, var2, 'Arrays of file and data batching not equal!')

    shutil.rmtree(base_path)


def test_multitarget_processor():
    classes = 20
    batch_size = 17
    max_number_of_classes_per_sample = 10
    processor = TargetIdx2MultiTarget(classes, 'test', 'test_transformed')
    for i in range(1):
        t = np.random.randint(1, classes, size=(batch_size, 10))
        idx = np.random.randint(1, 10, size=(batch_size,))
        for i in range(batch_size):
            t[i,idx[i]:] = 0
        str2var = {'test' : t}
    str2var = processor.at_batch_prepared(str2var)
    assert str2var['test_transformed'].shape == (batch_size, classes), 'Shape of test_transformed incorrect'


    expected = np.zeros((17, classes))
    for i, row in enumerate(t):
        for col in row:
            if col == 0: break
            expected[i, col] = 1

    np.testing.assert_array_equal(expected, str2var['test_transformed'], 'TargetIdx2MultiTarget indexing not working correctly')


test_data = [(lambda x: x.ent_type_, NERTokenizer),
             (lambda x: x.pos_, POSTokenizer),
             (lambda x: x.dep_, DependencyParser)]
ids = ['NER', 'POS', 'DEP']
@pytest.mark.parametrize("spacy_func, class_value", test_data, ids=ids)
def test_spacy_tokenization(spacy_func, class_value):
    nlp = spacy.load('en')
    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli10'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_sent_processor(class_value())
    p.add_sent_processor(SaveStateToList('tokens'))
    state = p.execute(s)

    inp_sents = state['data']['tokens']['input']
    sup_sents = state['data']['tokens']['support']
    sents = inp_sents + sup_sents

    # 2. setup nltk tokenization
    with open(get_test_data_path_dict()['snli10']) as f:
        tokenized_sents = {'input' : [], 'support' : []}
        for line in f:
            inp, sup, t = json.loads(line)
            tokenized_sents['input'].append([ spacy_func(token) for token in nlp(inp)])
            tokenized_sents['support'].append([ spacy_func(token) for token in nlp(sup)])

    sents_nltk = tokenized_sents['input'] + tokenized_sents['support']
    # 3. test equality
    assert len(sents) == len(sents_nltk), 'Sentence count differs!'
    log.debug('count should be 200: {0}', len(sents))
    for sent1, sent2 in zip(sents, sents_nltk):
        assert len(sent1) == len(sent2), 'Entity count differs!'
        for token1, token2 in zip(sent1, sent2):
            assert token1 == token2, 'Entity token values differ!'

def test_tfidf():
    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_sent_processor(TfidfFitter())
    p.add_sent_processor(ToLower())
    p.add_sent_processor(Tokenizer())
    p.add_sent_processor(SaveStateToList('tokens'))
    p.add_sent_processor(TfidfTransformer())
    p.add_sent_processor(SaveStateToList('tfidf'))
    state = p.execute(s)


    inp_tokens = state['data']['tokens']['input']
    sup_tokens = state['data']['tokens']['support']
    tfidf_inp_tokens = state['data']['tfidf']['input']
    tfidf_sup_tokens = state['data']['tfidf']['support']

    # 2. setup tfidf vectorization
    sup_tfidf = TfidfVectorizer()
    inp_tfidf = TfidfVectorizer()
    inps = []
    sups = []

    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : []}
        for line in f:
            inp, sup, t = json.loads(line)
            inps.append(inp)
            sups.append(sup)

    sups_transformed = sup_tfidf.fit_transform(sups)
    inps_transformed = inp_tfidf.fit_transform(inps)

    weighted_inp_docs = []
    weighted_sup_docs = []
    for i, (doc1, doc2) in enumerate(zip(inp_tokens, sup_tokens)):
        weights = []
        for token in doc1:
            if token in inp_tfidf.vocabulary_:
                weights.append(inps_transformed[i, inp_tfidf.vocabulary_[token]])
            else:
                weights.append(0.0)
        weighted_inp_docs.append(weights)

        weights = []
        for token in doc2:
            if token in sup_tfidf.vocabulary_:
                weights.append(sups_transformed[i, sup_tfidf.vocabulary_[token]])
            else:
                weights.append(0.0)
        weighted_sup_docs.append(weights)

    assert len(weighted_inp_docs) == len(tfidf_inp_tokens), 'Different amount of "documents"'
    for doc1, doc2, in zip(weighted_inp_docs, tfidf_inp_tokens):
        assert len(doc1) == len(doc2), 'Documents have differernt length!'
        for num1, num2 in zip(doc1, doc2):
            assert np.round(num1, 7) == np.round(num2, 7), 'Tfidf values are different!'

    assert len(weighted_sup_docs) == len(tfidf_sup_tokens), 'Different amount of "documents"'
    for doc1, doc2, in zip(weighted_sup_docs, tfidf_sup_tokens):
        assert len(doc1) == len(doc2), 'Documents have differernt length!'
        for num1, num2 in zip(doc1, doc2):
            assert np.round(num1, 7) == np.round(num2, 7), 'Tfidf values are different!'




test_data = [17, 128]
ids = ['batch_size=17', 'batch_size=128']
@pytest.mark.parametrize("batch_size", test_data, ids=ids)
def test_batch_streaming_pipeline(batch_size):
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    pipeline_folder = 'test_pipeline'

    s = DatasetStreamer()
    s.set_path(get_test_data_path_dict()['snli1k'])
    s.add_stream_processor(JsonLoaderProcessors())
    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_token_processor(AddToVocab())
    p.add_sent_processor(CustomTokenizer(tokenizer.tokenize))
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    state = p.execute(s)

    # 2. Load data from the SaveStateToList hook
    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['support']
    t_indices = state['data']['idx']['target']
    max_inp_len = np.max(state['data']['lengths']['input'])
    max_sup_len = np.max(state['data']['lengths']['support'])
    # For SNLI the targets consist of single words'

    # 3. parse data to numpy
    n = len(inp_indices)
    X = np.zeros((n, max_inp_len), dtype=np.int64)
    X_len = np.zeros((n), dtype=np.int64)
    S = np.zeros((n, max_sup_len), dtype=np.int64)
    S_len = np.zeros((n), dtype=np.int64)
    T = np.zeros((n, 1), dtype=np.int64)

    for i in range(len(inp_indices)):
        sample_inp = inp_indices[i][0]
        sample_sup = sup_indices[i][0]
        sample_t = t_indices[i][0]
        l = len(sample_inp)
        X_len[i] = l
        X[i, :l] = sample_inp

        l = len(sample_sup)
        S_len[i] = l
        S[i, :l] = sample_sup

        T[i] = sample_t[0]

    epochs = 2

    # 4. test data equality
    str2var_expected = {}
    str2var_expected['input'] = X
    str2var_expected['support'] = S
    str2var_expected['target'] = T
    str2var_expected['input_length'] = X_len
    str2var_expected['support_length'] = S_len
    for epoch in range(epochs):
        for i, str2var in enumerate(p.stream(s, batch_size)):
            assert np.int32 == str2var['input'].dtype, 'Input length type should be int32!'
            assert np.int32 == str2var['support'].dtype, 'Support length type should be int32!'
            assert np.int32 == str2var['target'].dtype, 'Target type should be int32!'
            assert np.int32 == str2var['index'].dtype, 'Index type should be int32!'
            idx = str2var['index']
            np.testing.assert_array_equal(X_len[idx], str2var['input_length'], 'Input length data not equal!')
            np.testing.assert_array_equal(S_len[idx], str2var['support_length'], 'Support length data not equal!')
            np.testing.assert_array_equal(X[idx], str2var['input'], 'Input data not equal!')
            np.testing.assert_array_equal(S[idx], str2var['support'], 'Support data not equal!')
            np.testing.assert_array_equal(T[idx], str2var['target'], 'Target data not equal!')
