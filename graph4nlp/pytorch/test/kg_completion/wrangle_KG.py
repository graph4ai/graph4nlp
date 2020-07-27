from __future__ import print_function
from os.path import join
import json

import argparse
import datetime
import json
import urllib
import pickle
import os
import numpy as np
import operator
import sys

rdm = np.random.RandomState(234234)

if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
else:
    dataset_name = 'FB15k-237'
    #dataset_name = 'FB15k'
    #dataset_name = 'yago'
    #dataset_name = 'WN18RR'

print('Processing dataset {0}'.format(dataset_name))

rdm = np.random.RandomState(2342423)
base_path = 'data/{0}/'.format(dataset_name)
files = ['train.txt', 'valid.txt', 'test.txt']

data = []
for p in files:
    with open(join(base_path, p)) as f:
        data = f.readlines() + data


label_graph = {}
train_graph = {}
test_cases = {}
for p in files:
    test_cases[p] = []
    train_graph[p] = {}


for p in files:
    with open(join(base_path, p)) as f:
        for i, line in enumerate(f):
            e1, rel, e2 = line.split('\t')
            e1 = e1.strip()
            e2 = e2.strip()
            rel = rel.strip()
            rel_reverse = rel+ '_reverse'

            # data
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)

            if (e1 , rel) not in label_graph:
                label_graph[(e1, rel)] = set()

            if (e2,  rel_reverse) not in label_graph:
                label_graph[(e2, rel_reverse)] = set()

            if (e1,  rel) not in train_graph[p]:
                train_graph[p][(e1, rel)] = set()
            if (e2, rel_reverse) not in train_graph[p]:
                train_graph[p][(e2, rel_reverse)] = set()

            # labels
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
            # (John, fatherOf_reverse, Mike)
            # (Tom, fatherOf_reverse, Mike)
            label_graph[(e1, rel)].add(e2)

            label_graph[(e2, rel_reverse)].add(e1)

            # test cases
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
            test_cases[p].append([e1, rel, e2])

            # data
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
            # (John, fatherOf_reverse, Mike)
            # (Tom, fatherOf_reverse, John)
            train_graph[p][(e1, rel)].add(e2)
            train_graph[p][(e2, rel_reverse)].add(e1)



def write_training_graph(cases, graph, path):
    with open(path, 'w') as f:
        n = len(graph)
        for i, key in enumerate(graph):
            e1, rel = key
            # (Mike, fatherOf, John)
            # (John, fatherOf, Tom)
            # (John, fatherOf_reverse, Mike)
            # (Tom, fatherOf_reverse, John)

            # (John, fatherOf) -> Tom
            # (John, fatherOf_reverse, Mike) 
            entities1 = " ".join(list(graph[key]))

            data_point = {}
            data_point['e1'] = e1
            data_point['e2'] = 'None'
            data_point['rel'] = rel
            data_point['rel_eval'] = 'None'
            data_point['e2_multi1'] =  entities1
            data_point['e2_multi2'] = "None"

            f.write(json.dumps(data_point)  + '\n')

def write_evaluation_graph(cases, graph, path):
    with open(path, 'w') as f:
        n = len(cases)
        n1 = 0
        n2 = 0
        for i, (e1, rel, e2) in enumerate(cases):
            # (Mike, fatherOf) -> John
            # (John, fatherOf, Tom)
            rel_reverse = rel+'_reverse'
            entities1 = " ".join(list(graph[(e1, rel)]))
            entities2 = " ".join(list(graph[(e2, rel_reverse)]))

            n1 += len(entities1.split(' '))
            n2 += len(entities2.split(' '))


            data_point = {}
            data_point['e1'] = e1
            data_point['e2'] = e2
            data_point['rel'] = rel
            data_point['rel_eval'] = rel_reverse
            data_point['e2_multi1'] = entities1
            data_point['e2_multi2'] = entities2

            f.write(json.dumps(data_point)  + '\n')


all_cases = test_cases['train.txt'] + test_cases['valid.txt'] + test_cases['test.txt']
write_training_graph(test_cases['train.txt'], train_graph['train.txt'], 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name))
write_evaluation_graph(test_cases['valid.txt'], label_graph, join('data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)))
write_evaluation_graph(test_cases['test.txt'], label_graph, 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name))
write_training_graph(all_cases, label_graph, 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name))
