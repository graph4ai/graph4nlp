import os
import json
import argparse
from collections import defaultdict


def parse_dataset(file_name):
    # Regard edges as additional nodes
    dataset = []
    with open(file_name, 'r') as f:
        prev_id = 0
        edge_count = 0
        out_obj = {}
        g_adj = defaultdict(list)
        g_ids_features_reversed = {}
        start_id, end_id = -1, -1

        for line in f:
            tokens = line.split()
            line_id = int(tokens[0])
            if line_id < prev_id:
                g_ids_features = {v: k for k, v in g_ids_features_reversed.items()}
                assert start_id in g_ids_features and end_id in g_ids_features
                assert 'seq' in out_obj

                for k in g_ids_features:
                    if k == start_id:
                        g_ids_features[k] = 'START'
                    if k == end_id:
                        g_ids_features[k] = 'END'

                    if len(g_ids_features[k].split('-')) > 1:
                        g_ids_features[k] = g_ids_features[k].split('-')[0]

                g_ids = dict(zip(range(len(g_ids_features)), range(len(g_ids_features))))
                out_obj['g_ids'] = g_ids
                out_obj['g_ids_features'] = g_ids_features
                out_obj['g_adj'] = g_adj
                dataset.append(out_obj)

                out_obj = {}
                g_adj = defaultdict(list)
                g_ids_features_reversed = {}
                edge_count = 0
                start_id, end_id = -1, -1

            if len(tokens) == 4:
                # edge line
                src = tokens[1]
                etype = tokens[2]
                tgt = tokens[3]

                if src not in g_ids_features_reversed:
                    src_id = len(g_ids_features_reversed)
                    g_ids_features_reversed[src] = src_id
                else:
                    src_id = g_ids_features_reversed[src]

                if tgt not in g_ids_features_reversed:
                    tgt_id = len(g_ids_features_reversed)
                    g_ids_features_reversed[tgt] = tgt_id
                else:
                    tgt_id = g_ids_features_reversed[tgt]

                edge_id = len(g_ids_features_reversed)
                g_ids_features_reversed['{}-{}'.format(etype, edge_count)] = edge_id
                edge_count += 1

                g_adj[tgt_id].append(edge_id)
                g_adj[edge_id].append(src_id)
            else:
                # question line
                # path question, task 19
                assert tokens[2] == 'path'
                src = tokens[3]
                tgt = tokens[4]
                label_str = tokens[5]
                labels = label_str.split(',')
                out_obj['seq'] = ' '.join(labels)

                start_id = g_ids_features_reversed[src]
                end_id = g_ids_features_reversed[tgt]

            prev_id = line_id

    return dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='../../../data/babi-19', help='Path to the directory that contains generated raw symbolic data, should contain two directories train, dev  and test.')
    parser.add_argument('-o', '--output_dir', default='../../../data/babi-19', help='Path to the directory to store processed symbolic data.')

    opt = vars(parser.parse_args())


    for file in ('train', 'dev', 'test'):
        d = parse_dataset(os.path.join(opt['input_dir'], '{}.txt'.format(file)))
        with open(os.path.join(opt['output_dir'], '{}.ndjson'.format(file)), 'w') as outf:
            for each in d:
                outf.write(json.dumps(each) + '\n')
