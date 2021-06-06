from __future__ import unicode_literals
from os.path import join
from spodernet.utils.util import Timer
from spodernet.utils.util import get_data_path, save_data, make_dirs_if_not_exists, load_data, Timer
from spodernet.interfaces import IAtBatchPreparedObservable
from spodernet.utils.global_config import Config
from past.builtins import basestring, long

import numpy as np
import os
import copy
import spacy
import nltk
import json
import pickle

from spodernet.utils.logger import Logger
log = Logger('processors.py.txt')

nlp = spacy.load('en_core_web_sm')
timer = Timer()

class KeyToKeyMapper(IAtBatchPreparedObservable):
    def __init__(self, key2key):
        self.key2key = key2key

    def at_batch_prepared(self, batch_parts):
        str2var = batch_parts
        new_str2var = {}
        for key1, key2 in self.key2key.items():
            new_str2var[key2] = str2var[key1]

        return new_str2var

class DictConverter(IAtBatchPreparedObservable):
    def __init__(self, keys=['input', 'support', 'target']):
        self.keys = keys

    def at_batch_prepared(self, batch_parts):
        str2var = {}
        i = 0
        for key in self.keys:
            str2var[key] = batch_parts[i]
            i += 1
            if i == 2*len(self.keys): break
            str2var[key+'_length'] = batch_parts[i]
            i += 1

        str2var['index'] = batch_parts[-1]

        return str2var


class TargetIdx2MultiTarget(IAtBatchPreparedObservable):
    def __init__(self, num_labels, variable_name, new_variable_name, shape=None, stop_index=0):
        self.num_labels = num_labels
        self.variable_name = variable_name
        self.new_variable_name = new_variable_name
        self.shape = shape
        self.stop_index = stop_index


    def at_batch_prepared(self, str2var):
        t = str2var[self.variable_name]
        if self.shape:
            new_t = np.zeros(self.shape, dtype=np.int64)
        else:
            new_t = np.zeros((t.shape[0], self.num_labels), dtype=np.int64)
        is_packed_array = isinstance(t[0], np.ndarray)

        for i, row in enumerate(t):
            if (isinstance(t, list) or len(t.shape) == 1):
                if row == self.stop_index: continue
                new_t[i, row] = 1
            else:
                for col in row:
                    if col == self.stop_index: break
                    new_t[i, col] = 1

        str2var[self.new_variable_name] = new_t

        return str2var

class VariableLengthSorter(IAtBatchPreparedObservable):
    def __init__(self, variable_name, postfix):
        self.variable_name = variable_name
        self.postfix = postfix

    def at_batch_prepared(self, str2var):
        var_len = str2var[self.variable_name + '_length']
        argidx = np.argsort(var_len)[::-1]

        for key in str2var.keys():
            str2var[key+self.postfix] = str2var[key][argidx]
            if 'length' in key:
                str2var[key+self.postfix] = str2var[key][argidx].tolist()

        return str2var

class ListIndexRemapper(object):
    def __init__(self, list_of_new_idx):
        self.list_of_new_idx = list_of_new_idx

    def at_batch_prepared(self, line):
        new_line = []
        for idx in self.list_of_new_idx:
            new_line.append(line[idx])

        return new_line

class JsonLoaderProcessors(object):
    def process(self, line):
        return json.loads(line)

class RemoveLineOnJsonValueCondition(object):
    def __init__(self, key, func_condition):
        self.key = key
        self.func_condition = func_condition

    def process(self, json_dict):
        if self.func_condition(json_dict[self.key]):
            return None
        else:
            return json_dict

class DictKey2ListMapper(object):
    def __init__(self, ordered_keys_source):
        self.ordered_keys_source = ordered_keys_source

    def process(self, dict_object):
        list_of_ordered_values = []
        for key in self.ordered_keys_source:
            list_of_ordered_values.append(dict_object[key])
        return list_of_ordered_values


class AbstractProcessor(object):
    def __init__(self):
        self.state = None
        self.execution_state = set(['fit', 'transform'])
        self.sample_counter = 0
        self.timer = Timer(silent=True)

    def link_with_pipeline(self, state):
        self.state = state

    def abstract_process(self, inputs, inp_type, benchmark):
        benchmark=True
        if benchmark:
            self.sample_counter +=1
            self.timer.tick()
        result = self.process(inputs, inp_type)
        if benchmark:
            self.timer.tick()
            if self.sample_counter == 10000:
                log.info_once('Time taken for 10000 samples for input type {0} for processor {1}: '.format(inp_type, type(self).__name__) + '{0} seconds', round(self.timer.tock(), 2))
        return result

    def process(self, inputs, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractProcessor need to implement the process method')


class AbstractLoopLevelTokenProcessor(AbstractProcessor):
    def __init__(self):
        super(AbstractLoopLevelTokenProcessor, self).__init__()
        self.successive_for_loops_to_tokens = None
        self.execution_state = set(['fit', 'transform'])

    def process_token(self, token, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractLoopLevelTokenProcessor need to implement the process_token method ')

    def process(self, sample, inp_type):
        if self.successive_for_loops_to_tokens == None:
            i = 0
            level = sample
            while not (   isinstance(level, basestring)
                       or isinstance(level, long)):
                    level = level[0]
                    i+=1
            self.successive_for_loops_to_tokens = i

        if self.successive_for_loops_to_tokens == 0:
            ret = self.process_token(sample, inp_type)

        elif self.successive_for_loops_to_tokens == 1:
            new_tokens = []
            for token in sample:
                new_tokens.append(self.process_token(token, inp_type))
            ret = new_tokens

        elif self.successive_for_loops_to_tokens == 2:
            new_sents = []
            for sent in sample:
                new_tokens = []
                for token in sent:
                    new_tokens.append(self.process_token(token, inp_type))
                new_sents.append(new_tokens)
            ret = new_sents

        return ret

class AbstractLoopLevelListOfTokensProcessor(AbstractProcessor):
    def __init__(self):
        super(AbstractLoopLevelListOfTokensProcessor, self).__init__()
        self.successive_for_loops_to_list_of_tokens = None
        self.execution_state = set(['fit', 'transform'])

    def process_list_of_tokens(self, tokens, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractLoopLevelListOfTokensProcessor need to implement the process_list_of_tokens method ')

    def process(self, sample, inp_type):
        if self.successive_for_loops_to_list_of_tokens == None:
            i = 0
            level = sample
            while not (isinstance(level, basestring)
                       or isinstance(level, int)
                       or isinstance(level, np.int32)
                       or isinstance(level, np.float32)):
                    level = level[0]
                    i+=1
            self.successive_for_loops_to_list_of_tokens = i-1

        if self.successive_for_loops_to_list_of_tokens == 0:
            ret = self.process_list_of_tokens(sample, inp_type, samples_idx)

        elif self.successive_for_loops_to_list_of_tokens == 1:
            new_sents = []
            for sent in sample:
                new_sents.append(self.process_list_of_tokens(sent, inp_type))
            ret = new_sents

        return ret

class TfidfFitter(AbstractProcessor):
    def __init__(self):
        super(TfidfFitter, self).__init__()
        self.execution_state = set(['fit'])

    def link_with_pipeline(self, state):
        self.tfidf = state['tfidf']
        state['tfidf_data'] = {}
        self.data = state['tfidf_data']

    def process(self, data, inp_type):
        if inp_type not in self.data: self.data[inp_type] = []
        self.data[inp_type].append(data)
        return data

class TfidfTransformer(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self):
        super(TfidfTransformer, self).__init__()
        self.fitted = set()
        self.execution_state = set(['transform'])

    def link_with_pipeline(self, state):
        self.tfidf = state['tfidf']
        self.data = state['tfidf_data']

    def process(self, list_of_token, inp_type):
        if inp_type not in self.fitted:
            self.tfidf[inp_type].fit(self.data[inp_type])
            self.fitted.add(inp_type)
        doc = ' '.join(list_of_token)
        X = self.tfidf[inp_type].transform([doc])
        weights = []
        vocab = self.tfidf[inp_type].vocabulary_
        for token in list_of_token:
            if token in vocab:
                idx = self.tfidf[inp_type].vocabulary_[token]
                weights.append(X[0, idx])
            else:
                weights.append(0.0)

        return weights

class DeepSeqMap(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, func):
        super(DeepSeqMap, self).__init__()
        self.func = func

    def process_list_of_tokens(self, data, inp_type):
        return self.func(data)

class Tokenizer(AbstractProcessor):
    def __init__(self):
        super(Tokenizer, self).__init__()
        self.tokenizer = nltk.tokenize.WordPunctTokenizer()

    def process(self, sentence, inp_type):
        return self.tokenizer.tokenize(sentence)

class NERTokenizer(AbstractProcessor):
    def __init__(self):
        super(NERTokenizer, self).__init__()
        self.execution_state = set(['transform'])

    def process(self, sentence, inp_type):
        return [token.ent_type_ for token in nlp(sentence, disable=['parse'])]

class DependencyParser(AbstractProcessor):
    def __init__(self):
        super(DependencyParser, self).__init__()
        self.execution_state = set(['transform'])

    def process(self, sentence, inp_type):
        return [token.dep_ for token in nlp(sentence)]

class POSTokenizer(AbstractProcessor):
    def __init__(self):
        super(POSTokenizer, self).__init__()
        self.execution_state = set(['transform'])

    def process(self, sentence, inp_type):
        return [token.pos_ for token in nlp(sentence, disable=['parse, entity'])]

class SentTokenizer(AbstractProcessor):
    def __init__(self):
        super(SentTokenizer, self).__init__()

    def process(self, sentence, inp_type):
        return [sent.text.replace('\n', '') for sent in nlp(sentence, disable=['entity']).sents]

class CustomTokenizer(AbstractProcessor):
    def __init__(self, tokenizer_method):
        super(CustomTokenizer, self).__init__()
        self.tokenize = tokenizer_method

    def process(self, sentence, inp_type):
        return self.tokenize(sentence)

class NaiveNCharTokenizer(AbstractProcessor):
    def __init__(self, N=3):
        super(NaiveNCharTokenizer, self).__init__()
        self.N = N

    def process(self, sentence, inp_type):
        return [sentence[i:i+self.N] for i in range(0, len(sentence), self.N)]

class AddToVocab(AbstractLoopLevelTokenProcessor):
    def __init__(self, general_vocab_keys=['input', 'support']):
        super(AddToVocab, self).__init__()
        self.general_vocab_keys = set(general_vocab_keys)
        self.execution_state = set(['fit'])

    def process_token(self, token, inp_type):
        if inp_type == 'target':
            self.state['vocab']['general'].add_label(token)
            log.statistical('Example vocab target token {0}', 0.01, token)
        if inp_type in self.general_vocab_keys:
            self.state['vocab']['general'].add_token(token)
            message = 'Example vocab {0} token'.format(inp_type)
            log.statistical(message + ': {0}', 0.01, token)
        self.state['vocab'][inp_type].add_token(token)
        return token

class ToLower(AbstractProcessor):
    def __init__(self, exclude_keys=None):
        super(ToLower, self).__init__()
        self.exclude_keys = exclude_keys

    def process(self, token, inp_type):
        if self.exclude_keys is not None:
            if inp_type in self.exclude_keys:
                return token

        return token.lower()


class ConvertTokenToIdx(AbstractLoopLevelTokenProcessor):
    def __init__(self, keys2keys=None):
        super(ConvertTokenToIdx, self).__init__()
        self.keys2keys = keys2keys #maps key to other key, for example encode inputs with support vocabulary
        self.execution_state = set(['transform'])

    def process_token(self, token, inp_type):
        if not self.keys2keys is None and inp_type in self.keys2keys:
            return self.state['vocab'][self.keys2keys[inp_type]].get_idx(token)
        else:
            if inp_type != 'target':
                log.statistical('a non-label token {0}', 0.00001, token)
                return self.state['vocab']['general'].get_idx(token)
            else:
                log.statistical('a token {0}', 0.00001, token)
                return self.state['vocab']['general'].get_idx_label(token)

class ApplyFunction(AbstractProcessor):
    def __init__(self, func):
        super(ApplyFunction, self).__init__()
        self.func = func
        self.execution_state =['fit', 'transform']

    def process(self, data, inp_type):
        return self.func(data)

class SaveStateToList(AbstractProcessor):
    def __init__(self, name):
        super(SaveStateToList, self).__init__()
        self.name = name
        self.execution_state = set(['transform'])

    def link_with_pipeline(self, state):
        self.state = state
        if self.name not in self.state['data']:
            self.state['data'][self.name] = {}
        self.data = self.state['data'][self.name]

    def process(self, data, inp_type):
        if inp_type not in self.data: self.data[inp_type] = []
        self.data[inp_type].append(data)
        return data

class SaveLengthsToState(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self):
        super(SaveLengthsToState, self).__init__()
        self.execution_state = set(['fit'])

    def link_with_pipeline(self, state):
        self.state = state
        self.state['data']['lengths'] = {}
        self.data = self.state['data']['lengths']

    def process_list_of_tokens(self, tokens, inp_type):
        if inp_type not in self.data: self.data[inp_type] = []
        self.data[inp_type].append(int(len(tokens)))
        log.statistical('A list of tokens: {0}', 0.0001, tokens)
        log.debug_once('Pipeline {1}: A list of tokens: {0}', tokens, self.state['name'])
        return tokens

class Idx2MultiTargetConverter(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, num_labels, stop_index=0):
        super(Idx2MultiTargetConverter, self).__init__()
        self.num_labels = num_labels
        self.stop_index = stop_index
        self.execution_state = set(['transform'])

    def process_list_of_tokens(self, tokens, inp_type):
        out = [0]*self.num_labels
        for col in tokens:
            if col == self.stop_index: break
            out[col] = 1

        return out

class SaveMaxLengthsToState(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self):
        super(SaveMaxLengthsToState, self).__init__()
        self.execution_state = set(['fit'])

    def link_with_pipeline(self, state):
        self.state = state
        self.state['data']['max_lengths'] = {}
        self.data = self.state['data']['max_lengths']

    def process_list_of_tokens(self, tokens, inp_type):
        if inp_type not in self.data: self.data[inp_type] = 0
        self.data[inp_type] = max(self.data[inp_type], len(tokens))
        return tokens

class StreamToHDF5(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, name, samples_per_file=50000, keys=['input', 'support', 'target']):
        super(StreamToHDF5, self).__init__()
        self.execution_state = set(['transform'])
        self.max_length = None
        self.samples_per_file = samples_per_file
        self.name = name
        self.keys = copy.deepcopy(keys)
        if 'index' not in self.keys:
            self.keys.append('index')
        self.shard_id = {}
        self.max_lengths = {}
        self.data = {}
        self.datatypes = {}
        self.lengths = {}
        self.current_sample = {}
        self.idx = {}
        for key in self.keys:
            self.shard_id[key] = 0
            self.max_lengths[key] = 0
            self.data[key] = []
            self.datatypes[key] = None
            self.current_sample[key] = 0
            self.idx[key] = 0

        self.num_samples = None
        self.config = {'paths' : [], 'sample_count' : []}
        self.checked_for_lengths = False
        self.paths = {}
        self.shuffle_idx = None
        self.current_X = {}

    def link_with_pipeline(self, state):
        self.state = state
        self.base_path = join(self.state['path'], self.name)
        make_dirs_if_not_exists(self.base_path)

    def init_and_checks(self):
        if 'lengths' not in self.state['data']:
            log.error('Do a first pass to produce lengths first, that is use the "SaveLengths" ' \
                       'processor, execute, clean processors, then rerun the pipeline with hdf5 streaming.')
        if self.num_samples == None:
            self.num_samples = len(self.state['data']['lengths'][self.keys[0]])
        log.debug('Using type int32 for inputs and supports for now, but this may not be correct in the future')
        self.checked_for_lengths = True
        self.num_samples = len(self.state['data']['lengths'][self.keys[0]])
        log.debug('Number of samples as calcualted with the length data (SaveLengthsToState): {0}', self.num_samples)

    def process_list_of_tokens(self, tokens, inp_type):
        if not self.checked_for_lengths:
            self.init_and_checks()

        if self.datatypes[inp_type] is None:
            if isinstance(tokens[0], float):
                self.datatypes[inp_type] = np.float32
            elif isinstance(tokens[0], int):
                self.datatypes[inp_type] = np.int32
            else:
                raise ValueError('Unsupported type: {0} for key {1}'.format(type(tokens[0]), inp_type))

        if self.max_lengths[inp_type] == 0:
            if 'max_lengths' in self.state['data']:
                max_length = self.state['data']['max_lengths'][inp_type]
            else:
                max_length = np.max(self.state['data']['lengths'][inp_type])
            log.debug('Calculated max length for input type {0} to be {1}', inp_type, max_length)
            self.max_lengths[inp_type] = max_length
            log.statistical('max length of the dataset: {0}', 0.0001, max_length)
        if inp_type not in self.current_X:
            self.current_X[inp_type] = np.zeros((self.samples_per_file, self.max_lengths[inp_type]), dtype=self.datatypes[inp_type])
            self.current_sample[inp_type] = 0
        self.current_X[inp_type][self.current_sample[inp_type], :len(tokens)] = tokens
        self.current_sample[inp_type] += 1

        if inp_type == self.keys[-2]:
            self.data['index'].append(self.idx[inp_type])
        self.idx[inp_type] += 1

        if (self.current_sample[inp_type] % self.samples_per_file == 0
           or self.idx[inp_type] == self.num_samples):
            if self.current_sample[inp_type] > 0:
                self.save_to_hdf5(inp_type)


        if self.idx[inp_type] % 10000 == 0:
            if self.idx[inp_type] % 50000 == 0:
                log.info('Processed {0} samples so far...', self.idx[inp_type])
            else:
                log.debug('Processed {0} samples so far...', self.idx[inp_type])

        if self.idx[inp_type] == self.num_samples:
            counts = np.array(self.config['sample_count'])
            log.debug('Counts for each shard: {0}'.format(counts))
            fractions = counts / np.float32(np.sum(counts))
            self.config['fractions'] = fractions.tolist()
            self.config['counts'] = counts.tolist()
            self.config['paths'] = []
            self.config['max_lengths'] = self.max_lengths
            for i in range(fractions.size):
                self.config['paths'].append(self.paths[i])

            pickle.dump(self.config, open(join(self.base_path, 'hdf5_config.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)

        return tokens

    def save_to_hdf5(self, inp_type):
        idx = self.shard_id[inp_type]
        if self.current_sample[inp_type] >= self.samples_per_file -1:
            X = self.current_X[inp_type]
        else:
            X = self.current_X[inp_type][:self.current_sample[inp_type]]
        file_name = inp_type + '_' + str(idx+1) + '.hdf5'

        if inp_type == 'input':
            #self.shuffle_idx = np.arange(X.shape[0])
            log.debug_once('First row of input data with shape {1} written to hdf5: {0}', X[0], X.shape)
            #X = X[self.shuffle_idx]
        log.debug('Writing hdf5 file for input type {0} to disk. Using index {1} and path {2}', inp_type, idx, join(self.base_path, file_name))
        log.debug('Writing hdf5 data. One sample row: {0}, shape: {1}, type: {2}', X[0], X.shape, X.dtype)
        save_data(join(self.base_path, file_name), X)
        if idx not in self.paths: self.paths[idx] = []
        self.paths[idx].append(join(self.base_path, file_name))

        if inp_type == self.keys[0]:
            log.statistical('Count of shard {0}; should be {1} most of the time'.format(X.shape[0], self.samples_per_file), 0.1)
            self.config['sample_count'].append(X.shape[0])

        if inp_type != self.keys[-2]:
            start = idx*self.samples_per_file
            end = (idx+1)*self.samples_per_file
            X_len = np.array(self.state['data']['lengths'][inp_type][start:end], dtype=np.int32)
            file_name_len = inp_type + '_lengths_' + str(idx+1) + '.hdf5'
            #X_len = X_len[self.shuffle_idx]
            save_data(join(self.base_path, file_name_len), X_len)
            self.paths[idx].append(join(self.base_path, file_name_len))
        else:
            start = idx*self.samples_per_file
            end = (idx+1)*self.samples_per_file
            X_len = np.array(self.state['data']['lengths'][inp_type][start:end], dtype=np.int32)
            file_name_len = inp_type + '_lengths_' + str(idx+1) + '.hdf5'
            #X_len = X_len[self.shuffle_idx]
            save_data(join(self.base_path, file_name_len), X_len)
            self.paths[idx].append(join(self.base_path, file_name_len))

            file_name_index = 'index_' + str(idx+1) + '.hdf5'
            index = np.arange(self.idx[inp_type] - X.shape[0], self.idx[inp_type], dtype=np.int32)
            #index = index[self.shuffle_idx]
            save_data(join(self.base_path, file_name_index), index)
            self.paths[idx].append(join(self.base_path, file_name_index))

        self.shard_id[inp_type] += 1
        self.current_X.pop(inp_type, None)
        self.current_sample[inp_type] = 0




class StreamToBatch(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, keys=['input', 'support', 'target'], seed=234234):
        super(StreamToBatch, self).__init__()
        self.execution_state = set(['transform'])
        self.str2var = {}
        self.str2samples = {}
        self.rdm = np.random.RandomState(seed)
        for key in keys:
            self.str2samples[key] = []

    def process_list_of_tokens(self, tokens, inp_type):
        self.str2samples[inp_type].append(tokens)
        return tokens

    def shuffle(self):
        idx = None
        for key in self.str2samples.keys():
            if idx is None:
                variable = self.str2samples[key]
                idx = np.arange(variable.shape[0])
                self.rdm.shuffle(idx)

            self.str2samples[key] = variable[idx]

    def get_data(self):
        for key, variable in self.str2samples.items():
            n = len(variable)
            lengths = [len(tokens) for tokens in variable]
            max_length = np.max(lengths)
            x = np.zeros((n, max_length))
            for row, (l, sample) in enumerate(zip(lengths, variable)):
                x[row,:l] = sample

            self.str2var[key] = x
            self.str2var[key + '_length'] = np.array(lengths)
        return self.str2var
