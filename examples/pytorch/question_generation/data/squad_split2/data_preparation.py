import sys
import json
import chardet

fout = open(sys.argv[2], 'w')


with open(sys.argv[1]) as dataset_file:
    dataset = json.load(dataset_file, encoding='utf-8')
    all_instances = []
    for instance in dataset:
        ID_num = None
        if 'id' in instance: ID_num = instance['id']

        src = instance['annotation1']['toks'] if 'annotation1' in instance else instance['text1']
        if src == "": continue

        tgt = instance['annotation2']['toks'] if 'annotation2' in instance else instance['text2']
        if tgt == "": continue

        extra_text = None
        if 'text3' in instance or 'annotation3' in instance:
            extra_text = instance['annotation3']['toks'] if 'annotation3' in instance else instance['text3']

        fout.write('{}\t{}\t{}\n'.format(src, extra_text, tgt))
