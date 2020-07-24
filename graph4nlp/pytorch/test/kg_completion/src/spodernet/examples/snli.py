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

from spodernet.frontend import Model, PairedBiDirectionalLSTM, SoftmaxCrossEntropy, Embedding, Trainer

Config.parse_argv(sys.argv)

np.set_printoptions(suppress=True)

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

    train_batcher.subscribe_to_events(AccuracyHook('Train', print_every_x_batches=1000))
    dev_batcher.subscribe_to_events(AccuracyHook('Dev', print_every_x_batches=1000))
    eta = ETAHook(print_every_x_batches=1000)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)

    model = Model()
    model.add(Embedding(128, vocab.num_embeddings))
    model.add(PairedBiDirectionalLSTM(128, hidden_size=256, variable_length=True, conditional_encoding=False))
    model.add(SoftmaxCrossEntropy(input_size=256*4, num_labels=3))


    t = Trainer(model)
    for i in range(10):
        t.train(train_batcher, epochs=1)
        t.evaluate(dev_batcher)


if __name__ == '__main__':
    main()
