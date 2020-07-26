from __future__ import print_function
from os.path import join

import os
import numpy as np
import itertools
import sys
import numpy as np


rdm = np.random.RandomState(345345)


if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
    threshold = float(sys.argv[2])
else:
    #dataset_name = 'FB15k-237'
    #dataset_name = 'YAGO3-10'
    #dataset_name = 'WN18'
    #dataset_name = 'FB15k'
    dataset_name = 'WN18'
    threshold = 0.8

print(threshold)

base_path = 'data/{0}/'.format(dataset_name)
files = ['train.txt', 'valid.txt', 'test.txt']

data = []
for p in files:
    with open(join(base_path, p)) as f:
        data = f.readlines() + data

e_set = set()
rel_set = set()
test_cases = {}
rel_to_tuple = {}
e1rel2e2 = {}
existing_triples = set()
rel2tuple_train = {}
for p in files:
    test_cases[p] = []


for p in files:
    with open(join(base_path, p)) as f:
        for i, line in enumerate(f):
            e1, rel, e2 = line.split('\t')
            e1 = e1.strip()
            e2 = e2.strip()
            rel = rel.strip()
            e_set.add(e1)
            e_set.add(e2)
            rel_set.add(rel)
            existing_triples.add((e1, rel, e2))

            if (e1, rel) not in e1rel2e2: e1rel2e2[(e1, rel)] = set()
            e1rel2e2[(e1, rel)].add(e2)

            if rel not in rel_to_tuple:
                rel_to_tuple[rel] = set()

            if rel not in rel2tuple_train:
                rel2tuple_train[rel] = set()

            rel_to_tuple[rel].add((e1,e2))
            test_cases[p].append([e1, rel, e2])
            if p == 'train.txt':
                rel2tuple_train[rel].add((e1, e2))


def check_for_reversible_relations(rel_to_tuple, threshold=0.80):
    rel2reversal_rel = {}
    pairs = set()
    for i, rel1 in enumerate(rel_to_tuple):
        if i % 100 == 0:
            print('Processed {0} relations...'.format(i))
        for rel2 in rel_to_tuple:
            tuples2 = rel2tuple_train[rel2]
            tuples1 = rel2tuple_train[rel1]
            # check if the entire set of (e1, e2) is contained in the set of the 
            # other relation, but in a reversed manner
            # that is ALL (e1, e2) -> (e2, e1) for rel 1 are contained in set entity tuple set of rel2 (and vice versa)
            # if this is true for ALL entities, that is the sets completely overlap, then add a rule that
            # (e1, rel1, e2) == (e2, rel2, e1)
            n1 = float(len(tuples1))
            n2 = float(len(tuples2))
            left =  np.sum([(e2,e1) in tuples2 for (e1,e2) in tuples1])/n1
            right =  np.sum([(e1,e2) in tuples1 for (e2,e1) in tuples2])/n2
            if left >= threshold or right >= threshold:
                print(left, right, rel1, rel2, n1, n2)
                rel2reversal_rel[rel1] = rel2
                rel2reversal_rel[rel2] = rel1
                if (rel2, rel1) not in pairs:
                    pairs.add((rel1, rel2))
                #print(rel1, rel2, left, right)
    return rel2reversal_rel, pairs

rel2reversal_rel, banned_pairs = check_for_reversible_relations(rel_to_tuple, threshold)

print(rel2reversal_rel)
print(len(rel2reversal_rel))
evaluate = True
if evaluate:
    all_cases = []
    rel2tuples = {}
    train_dev = test_cases['train.txt'] + test_cases['valid.txt']
    for e1, rel, e2 in train_dev:
        if rel not in rel2tuples: rel2tuples[rel] = set()
        rel2tuples[rel].add((e1, e2))
        if rel in rel2reversal_rel:
            rel2 = rel2reversal_rel[rel]
            if rel2 not in rel2tuples: rel2tuples[rel2] = set()
            rel2tuples[rel2].add((e2, e1))

    num_entities = len(e_set)
    ranks = []
    for i, (e1, rel, e2) in enumerate(test_cases['test.txt']):
        if i % 1000 == 0: print(i)
        ranks.append(0)
        ranks.append(0)
        if (e1, e2) in rel2tuples[rel]:
            ranks[-1] += 1
            ranks[-2] += 1
            for e2_neg in e_set:
                if (e1, rel, e2_neg) in existing_triples: continue
                if (e1, e2_neg) in rel2tuples[rel]:
                    ranks[-1] += 1

            for e1_neg in e_set:
                if (e1_neg, rel, e2) in existing_triples: continue
                if (e1_neg, e2) in rel2tuples[rel]:
                    ranks[-2] += 1
            ranks[-1] = rdm.randint(1, ranks[-1]+1)
            ranks[-2] = rdm.randint(1, ranks[-2]+1)
        else:
            existing_entities1=0
            existing_entities2=0
            for e2_neg in e_set:
                if (e1, rel, e2_neg) in existing_triples:
                    existing_entities1+=1
                    continue
                if (e1, e2_neg) in rel2tuples[rel]:
                    ranks[-1] += 1
            for e1_neg in e_set:
                if (e1_neg, rel, e2) in existing_triples:
                    existing_entities2+=1
                    continue
                if (e1_neg, e2) in rel2tuples[rel]:
                    ranks[-2] += 1
            ranks[-1] = rdm.randint(1,num_entities+1-existing_entities1)
            ranks[-2] = rdm.randint(1,num_entities+1-existing_entities2)
    n = float(len(ranks))
    print(n)
    ranks = np.array(ranks)
    for i in range(10):
        print('Hits@{0}: {1:.7}'.format(i+1, np.sum(ranks <= i+1)/n))
    print("MR: {0}".format(np.mean(ranks)))
    print("MRR: {0}".format(np.mean(1.0/ranks)))

print(threshold)
