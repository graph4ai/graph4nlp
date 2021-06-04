from os.path import join

import os
import shutil
import json
import zipfile
import numpy as np

from spodernet.preprocessing.vocab import Vocab
from spodernet.utils.util import Timer
from spodernet.preprocessing.processors import SaveLengthsToState
from sklearn.feature_extraction.text import TfidfVectorizer

from spodernet.utils.logger import Logger
log = Logger('pipeline.py.txt')

t = Timer()
class StreamMethods:
    files = 'FILES'
    data = 'DATA'

class DatasetStreamer(object):
    def __init__(self, input_keys=None, output_keys=None, stream_method=StreamMethods.files):
        self.stream_processors = []
        self.input_keys = input_keys or ['input', 'support', 'target']
        self.output_keys = output_keys
        self.paths = []
        self.output_keys = output_keys or self.input_keys
        self.stream_method = stream_method
        self.data = []

    def add_stream_processor(self, stream):
        self.stream_processors.append(stream)

    def set_paths(self, list_of_paths):
        self.paths = list_of_paths

    def set_path(self, path):
        self.set_paths([path])

    def set_data(self, data):
        self.data = [data]

    def stream_files(self):
        if self.stream_method == StreamMethods.files:
            stream_objects = [open(p) for p in self.paths]
        elif self.stream_method == StreamMethods.data:
            stream_objects = self.data
        else:
            raise Exception('Unrecognized streaming method')

        try:
            for obj in stream_objects:
                for line in obj:
                    filtered = False
                    for streamp in self.stream_processors:
                        line = streamp.process(line)
                        if line is None:
                            filtered = True
                            break
                    if filtered:
                        continue
                    else:
                        log.debug_once('First line processed by line processors: {0}', line)
                        data = []
                        inputkey2data = {}
                        for input_key, variable in zip(self.input_keys, line):
                            inputkey2data[input_key] = variable

                        for output_key in self.output_keys:
                            data.append(inputkey2data[output_key])

                        yield data
        except Exception as e:
            if self.stream_method == StreamMethods.files:
                for fh in stream_objects:
                    fh.close()
            raise

class Pipeline(object):
    def __init__(self, name, delete_all_previous_data=False, keys=None, skip_transformation=False, benchmark=False):
        self.keys = keys or ['input', 'support', 'target']
        home = os.environ['HOME']
        self.root = join(home, '.data', name)
        self.tfidf = TfidfVectorizer()
        self.skip_transformation = skip_transformation
        self.benchmark = benchmark

        if not os.path.exists(self.root):
            log.debug_once('Pipeline path {0} does not exist. Creating folder...', self.root)
            os.mkdir(self.root)
        else:
            if delete_all_previous_data:
                log.warning('delete_all_previous_data=True! Deleting all folder contents of folder {0}!', self.root)
                shutil.rmtree(self.root)
                log.info('Recreating path: {0}', self.root)
                os.mkdir(self.root)
            else:
                log.warning('Pipeline path {0} already exist. This pipeline may overwrite data in this path!', self.root)

        self.state = {'name' : name, 'home' : home, 'path' : self.root, 'data' : {}}
        self.state['vocab'] = {}
        self.state['tfidf'] = {}
        self.state['vocab']['general'] = Vocab(path=join(self.root, 'vocab'))
        self.state['tfidf']['general'] = TfidfVectorizer(stop_words=[])
        for key in self.keys:
            self.state['vocab'][key] = Vocab(path=join(self.root, 'vocab_'+key))
            self.state['tfidf'][key] = TfidfVectorizer(stop_words=[])

        self.text_processors = []
        self.sent_processors = []
        self.token_processors = []
        self.post_processors = [(self.keys, SaveLengthsToState())]
        self.post_processors[-1][1].link_with_pipeline(self.state)

    def add_text_processor(self, text_processor, keys=None):
        keys = keys or self.keys
        text_processor.link_with_pipeline(self.state)
        log.debug('Added text preprocessor {0}', type(text_processor))
        self.text_processors.append([keys, text_processor])

    def add_sent_processor(self, sent_processor, keys=None):
        keys = keys or self.keys
        sent_processor.link_with_pipeline(self.state)
        log.debug('Added sent preprocessor {0}', type(sent_processor))
        self.sent_processors.append([keys, sent_processor])

    def add_token_processor(self, token_processor, keys=None):
        keys = keys or self.keys
        token_processor.link_with_pipeline(self.state)
        log.debug('Added token preprocessor {0}', type(token_processor))
        self.token_processors.append([keys, token_processor])

    def add_post_processor(self, post_processor, keys=None):
        keys = keys or self.keys
        post_processor.link_with_pipeline(self.state)
        log.debug('Added post preprocessor {0}', type(post_processor))
        self.post_processors.append([keys, post_processor])


    def clear_processors(self):
        self.post_processors = [(self.keys, SaveLengthsToState())]
        self.post_processors[-1][1].link_with_pipeline(self.state)
        self.sent_processors = []
        self.token_processors = []
        self.text_processors = []
        log.debug('Cleared processors of pipeline {0}', self.state['name'])

    def clear_lengths(self):
        self.state['data'].pop('lengths', None)

    def save_vocabs(self):
        self.state['vocab']['general'].save_to_disk()
        for key in self.keys:
            self.state['vocab'][key].save_to_disk()

    def load_vocabs(self):
        loaded = True
        loaded = loaded and self.state['vocab']['general'].load_from_disk()
        for key in self.keys:
            loaded = loaded and self.state['vocab'][key].load_from_disk()
        return loaded

    def copy_vocab_from_pipeline(self, pipeline_or_vocab, vocab_type=None):
        if isinstance(pipeline_or_vocab, Pipeline):
            self.state['vocab'] = pipeline_or_vocab.state['vocab']
        elif isinstance(pipeline_or_vocab, Vocab):
            if vocab_type is None:
                self.state['vocab']['general'] = pipeline_or_vocab
            else:
                self.state['vocab'][vocab_type] = pipeline_or_vocab
        else:
            str_error = 'The add vocab method expects a Pipeline or Vocab instance as argument, got {0} instead!'.format(type(pipeline_or_vocab))
            log.error(str_error)
            raise TypeError(str_error)

    def iterate_over_processors(self, processors, variables):
        for filter_keys, textp in processors:
            for i, key in enumerate(self.keys):
                if key in filter_keys:
                    variables[i] = textp.abstract_process(variables[i], inp_type=key)
        return variables

    def execute(self, data_streamer):
        '''Tokenizes the data, calcs the max length, and creates a vocab.'''
        for execution_state in ['fit', 'transform']:
            if execution_state == 'tranform' and self.skip_transformation: return self.state
            for iter_count, var in enumerate(data_streamer.stream_files()):
                for filter_keys, textp in self.text_processors:
                    if execution_state not in textp.execution_state: continue
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            var[i] = textp.abstract_process(var[i], key, self.benchmark)

                for i in range(len(var)):
                    var[i] = (var[i] if isinstance(var[i], list) else [var[i]])

                for filter_keys, sentp in self.sent_processors:
                    if execution_state not in sentp.execution_state: continue
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            for j in range(len(var[i])):
                                var[i][j] = sentp.abstract_process(var[i][j], key, self.benchmark)

                for i in range(len(var)):
                    var[i] = (var[i] if isinstance(var[i][0], list) else [[sent] for sent in var[i]])

                for filter_keys, tokenp in self.token_processors:
                    if execution_state not in tokenp.execution_state: continue
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            for j in range(len(var[i])):
                                for k in range(len(var[i][j])):
                                    var[i][j][k] = tokenp.abstract_process(var[i][j][k], key, self.benchmark)

                for filter_keys, postp in self.post_processors:
                    if execution_state not in postp.execution_state: continue
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            var[i] = postp.abstract_process(var[i], key, self.benchmark)

        return self.state

    def stream(self, data_streamer, batch_size, skip_probability=0.0):
        str2var = {}
        key2max_len_and_type = {}
        index = 0
        for execution_state in ['fit', 'transform']:
            for iter_count, var in enumerate(data_streamer.stream_files()):
                for filter_keys, textp in self.text_processors:
                    if execution_state not in textp.execution_state: continue
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            var[i] = textp.abstract_process(var[i], key, self.benchmark)

                for i in range(len(var)):
                    var[i] = (var[i] if isinstance(var[i], list) else [var[i]])

                for filter_keys, sentp in self.sent_processors:
                    if execution_state not in sentp.execution_state: continue
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            for j in range(len(var[i])):
                                var[i][j] = sentp.abstract_process(var[i][j], key, self.benchmark)

                for i in range(len(var)):
                    var[i] = (var[i] if isinstance(var[i][0], list) else [[sent] for sent in var[i]])

                for filter_keys, tokenp in self.token_processors:
                    if execution_state not in tokenp.execution_state: continue
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            for j in range(len(var[i])):
                                for k in range(len(var[i][j])):
                                    var[i][j][k] = tokenp.abstract_process(var[i][j][k], key, self.benchmark)

                for filter_keys, postp in self.post_processors:
                    if execution_state not in postp.execution_state: continue
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            var[i] = postp.abstract_process(var[i], key, self.benchmark)
                if execution_state == 'transform':

                    for i, key in enumerate(self.keys):
                        if key not in str2var:
                            str2var[key] = []
                            str2var[key+'_length'] = []
                        str2var[key].append(var[i])
                        str2var[key+'_length'].append(self.state['data']['lengths'][key][index])
                    if 'index' not in str2var: str2var['index'] = []
                    str2var['index'].append(index)

                    if len(str2var[key]) == batch_size:
                        for key in self.keys:
                            batches = str2var[key]

                            if key not in key2max_len_and_type:
                                if isinstance(batches[0][0][0], float):
                                    dtype = np.float32
                                elif isinstance(batches[0][0][0], int):
                                    dtype = np.int32
                                else:
                                    raise Exception('Unknown data type: {0} for item {1}'.format(type(batches[0][0][0]), batches[0][0][0]))

                                max_len = np.max(self.state['data']['lengths'][key])
                                key2max_len_and_type[key] = (max_len, dtype)

                            empty_batch = np.zeros((batch_size, key2max_len_and_type[key][0]), dtype=key2max_len_and_type[key][1])
                            var = str2var[key]
                            lengths = str2var[key+'_length']
                            for row in range(len(var)):
                                data = var[row]
                                while isinstance(data, list):
                                    if not isinstance(data[0], list): break
                                    data = data[0]
                                empty_batch[row, :lengths[row]] = data
                            str2var[key] = empty_batch
                        str2var['index'] = np.array(str2var['index'], dtype=np.int32)
                        yield str2var
                        str2var = {}
                    index += 1
