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
        g['seq'] = t.strip().split('\t')[-1].lower()
        dataset.append(g)

    a = 0

    with open('../../../data/job_dataset/'+part+'.ndjson', 'w') as outf:
        for each in dataset:
            outf.write(json.dumps(each) + '\n')

create_njson('train','../../../data/job_dataset/graph.train','../../../data/job_dataset/train.txt')
create_njson('test','../../../data/job_dataset/graph.test','../../../data/job_dataset/test.txt')


