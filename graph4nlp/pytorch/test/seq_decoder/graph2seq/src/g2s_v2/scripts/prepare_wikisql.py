import argparse
import os
import json
from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize



##############
# Below copied from the WikiSQL code
agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']
##############



def construct_sql_graph(sql, table):
    '''Details are provided in Appendix B.3 in https://arxiv.org/abs/1804.00823'''
    assert len(sql) == 3
    adj_str = defaultdict(list)
    col = ' '.join(wordpunct_tokenize(table['header'][sql['sel']]))
    agg = agg_ops[sql['agg']]
    adj_str[col].append('select')
    if agg is not '':
        adj_str[col].append(agg)


    if len(sql['conds']) > 1:
        joint_node = 'and'
        adj_str['select'].append(joint_node)
    else:
        joint_node = 'select'

    for cond in sql['conds']:
        assert len(cond) == 3
        cond_col = ' '.join(wordpunct_tokenize(table['header'][cond[0]]))
        adj_str[cond_ops[cond[1]] + ' ' + ' '.join(wordpunct_tokenize(str(cond[2])))].append(cond_col)
        if joint_node == 'select':
            adj_str['select'].append(cond_col)
        else:
            adj_str[cond_col].append(joint_node)

    node_attrs = set(adj_str.keys())
    for each in adj_str.values():
        node_attrs.update(each)

    g_ids = dict(zip(range(len(node_attrs)), range(len(node_attrs))))
    g_ids_features = dict(zip(range(len(node_attrs)), node_attrs))
    g_ids_features_reversed = {v:k for k, v in g_ids_features.items()}

    g_adj = {}
    for n1, nodes in adj_str.items():
        g_adj[g_ids_features_reversed[n1]] = [g_ids_features_reversed[n2] for n2 in nodes]
    return g_ids, g_ids_features, g_adj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='path to the input file')
    parser.add_argument('-t', '--table', type=str, help='path to the input table')
    parser.add_argument('-o', '--output', required=True, type=str, help='path to the output file')

    opt = vars(parser.parse_args())

    db = {}
    with open(opt['table'], 'r') as intf:
        for line in intf:
            table = json.loads(line.strip())
            db[table['id']] = table

    with open(opt['output'], 'w') as outf:
        with open(opt['file'], 'r') as inf:
            for line in inf:
                qa = json.loads(line.strip())
                out_obj = {}
                out_obj['seq'] = ' '.join(wordpunct_tokenize(qa['question']))
                if not db.get(qa['table_id'], None):
                    continue

                out_obj['g_ids'], out_obj['g_ids_features'], out_obj['g_adj'] = construct_sql_graph(qa['sql'], db[qa['table_id']])
                outf.write(json.dumps(out_obj) + '\n')
