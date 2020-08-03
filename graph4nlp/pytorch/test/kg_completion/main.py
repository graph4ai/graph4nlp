import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
import math

from os.path import join
import torch.backends.cudnn as cudnn

from evaluation import ranking_and_hits
from models import SACN, ConvTransE, ConvE
from models_graph4nlp import DistMult, DistMultGNN, TransEGNN, TransE, Complex, ComplexGNN
from src.spodernet.spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from src.spodernet.spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from src.spodernet.spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from src.spodernet.spodernet.utils.global_config import Config, Backends
from src.spodernet.spodernet.utils.logger import Logger, LogLevel
from src.spodernet.spodernet.preprocessing.batching import StreamBatcher
from src.spodernet.spodernet.preprocessing.pipeline import Pipeline
from src.spodernet.spodernet.preprocessing.processors import TargetIdx2MultiTarget
from src.spodernet.spodernet.hooks import LossHook, ETAHook
from src.spodernet.spodernet.utils.util import Timer
from src.spodernet.spodernet.utils.cuda_utils import CUDATimer
from src.spodernet.spodernet.utils.cuda_utils import CUDATimer
from src.spodernet.spodernet.preprocessing.processors import TargetIdx2MultiTarget
import scipy.sparse as sp
import scipy
from os.path import join, exists
import os, sys
import pickle as pkl
import pickle
path_dir = os.getcwd()

np.set_printoptions(precision=3)

# timer = CUDATimer()
cudnn.benchmark = True

# parse console parameters and set global variables
Config.backend = Backends.TORCH
Config.parse_argv(sys.argv)
# Config.cuda = True
#Config.embedding_dim = 200

model_name = '{2}_{0}_{1}_{3}'.format(Config.input_dropout, Config.dropout, Config.model_name, Config.loss_name)
epochs = 1000
load = False
if Config.dataset is None:
    Config.dataset = 'FB15k-237'
model_path = 'saved_models/{0}_{1}.model'.format(Config.dataset, model_name)


''' Preprocess knowledge graph using spodernet. '''
def preprocess(dataset_name, delete_data=False):
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    keys2keys['rel_eval'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(Config.dataset, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys),
                         keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
    p.add_post_processor(StreamToHDF5('full', samples_per_file=1000, keys=input_keys))

    p.execute(d)
    p.save_vocabs()


    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)


def main():
    #config_path = join(path_dir, 'data', Config.dataset, 'data.npy')
    if Config.process: preprocess(Config.dataset, delete_data=True)
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(Config.dataset, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']
    node_list = p.state['vocab']['e1']
    rel_list = p.state['vocab']['rel']

    num_entities = vocab['e1'].num_token
    num_relations = vocab['rel'].num_token

    train_batcher = StreamBatcher(Config.dataset, 'train', Config.batch_size, randomize=True, keys=input_keys)
    dev_rank_batcher = StreamBatcher(Config.dataset, 'dev_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)
    test_rank_batcher = StreamBatcher(Config.dataset, 'test_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)

    train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))
    dev_rank_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))
    dev_rank_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi2', 'e2_multi2_binary'))
    test_rank_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))
    test_rank_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi2', 'e2_multi2_binary'))



    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    data = []
    rows = []
    columns = []

    for i, str2var in enumerate(train_batcher):
        print("batch number:", i)
        for j in range(str2var['e1'].shape[0]):
            for k in range(str2var['e2_multi1'][j].shape[0]):
                if str2var['e2_multi1'][j][k] != 0:
                    a = str2var['rel'][j].cpu()
                    data.append(str2var['rel'][j].cpu())
                    rows.append(str2var['e1'][j].cpu().tolist()[0])
                    columns.append(str2var['e2_multi1'][j][k].cpu())
                else:
                    break

    rows = rows  + [i for i in range(num_entities)]
    columns = columns + [i for i in range(num_entities)]
    data = data + [num_relations for i in range(num_entities)]

    if Config.cuda:
        indices = torch.LongTensor([rows, columns]).cuda()
        v = torch.LongTensor(data).cuda()
        adjacencies = [indices, v, num_entities]
    else:
        indices = torch.LongTensor([rows, columns])
        v = torch.LongTensor(data)
        adjacencies = [indices, v, num_entities]


    #filename = join(path_dir, 'data', Config.dataset, 'adj.pkl')
    #file = open(filename, 'wb+')
    #pkl.dump(adjacencies, file)
    #file.close()

    print('Finished the preprocessing')

    ############

    X = torch.LongTensor([i for i in range(num_entities)])

    if Config.model_name is None:
        model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'SACN':
        model = SACN(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'ConvTransE':
        model = ConvTransE(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'ConvE':
        model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'DistMult':
        model = DistMult(vocab['e1'].num_token, vocab['rel'].num_token, Config.loss_name)
    elif Config.model_name == 'DistMultGNN':
        model = DistMultGNN(vocab['e1'].num_token, vocab['rel'].num_token, Config.loss_name)
    elif Config.model_name == 'TransE':
        model = TransE(vocab['e1'].num_token, vocab['rel'].num_token, loss_name=Config.loss_name)
    elif Config.model_name == 'TransEGNN':
        model = TransEGNN(vocab['e1'].num_token, vocab['rel'].num_token, loss_name=Config.loss_name)
    elif Config.model_name == 'ComplEx':
        model = Complex(vocab['e1'].num_token, vocab['rel'].num_token, loss_name=Config.loss_name)
    elif Config.model_name == 'ComplExGNN':
        model = ComplexGNN(vocab['e1'].num_token, vocab['rel'].num_token, loss_name=Config.loss_name)
    else:
        # log.info('Unknown model: {0}', Config.model_name)
        raise Exception("Unknown model!")


    #train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))
    train_batcher = StreamBatcher(Config.dataset, 'train', Config.batch_size, randomize=True, keys=input_keys)

    eta = ETAHook('train', print_every_x_batches=100)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=100))
    train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))


    if Config.cuda:
        model.cuda()
        X = X.cuda()


    if load:
        model_params = torch.load(model_path)
        print(model)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        model.eval()
        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')
        ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
    else:
        model.init()

    total_param_size = []
    params = [value.numel() for value in model.parameters()]
    print(params)
    print(np.sum(params))

    opt = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.L2)
    for epoch in range(epochs):
        model.train()
        for i, str2var in enumerate(train_batcher):
            opt.zero_grad()
            if Config.cuda:
                e1 = str2var['e1'].cuda()
                rel = str2var['rel'].cuda()
                e2_multi = str2var['e2_multi1_binary'].float().cuda()
            else:
                e1 = str2var['e1']
                rel = str2var['rel']
                e2_multi = str2var['e2_multi1_binary'].float()
            if model.loss_name == "SoftMarginLoss":
                e2_multi[e2_multi==0] = -1
                pred = model.forward(e1, rel, X, adjacencies)
                loss = model.loss(pred, e2_multi)
            elif model.loss_name == "SoftplusLoss" or model.loss_name == "SigmoidLoss":
                pred, pos, neg = model.forward(e1, rel, X, adjacencies, e2_multi)
                loss = model.loss(pos, neg)
            else:
                # label smoothing
                e2_multi = ((1.0 - Config.label_smoothing_epsilon) * e2_multi) + (1.0 / e2_multi.size(1))
                pred = model.forward(e1, rel, X, adjacencies)
                loss = model.loss(pred, e2_multi)
            loss.backward()
            opt.step()

            train_batcher.state.loss = loss.cpu()


        print('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation', X, adjacencies)
            if epoch % 3 == 0:
                if epoch > 0:
                    ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation', X, adjacencies)


if __name__ == '__main__':
    main()

