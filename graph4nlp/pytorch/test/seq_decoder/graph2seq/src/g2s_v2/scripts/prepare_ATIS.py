import pickle as pkl
import json

def create_njson(part, graph_file, txt_file):
    dataset = []

    with open(txt_file,'r') as f:
        txt_data = f.readlines()
    with open(graph_file,'r') as f:
        graph_data = f.readlines()

    for t, g in zip(txt_data, graph_data):
        g = eval(g)
        g['seq'] = t.strip().split('\t')[-1]
        dataset.append(g)

    a = 0

    with open('../../../data/ATIS/'+part+'.ndjson', 'w') as outf:
        for each in dataset:
            outf.write(json.dumps(each) + '\n')

# create_njson('train','../../../data/ATIS/graph.train','../../../data/ATIS/train.txt')
# create_njson('test','../../../data/ATIS/graph.test','../../../data/ATIS/test.txt')
create_njson('valid','../../../data/ATIS/graph.valid','../../../data/ATIS/valid.txt')


