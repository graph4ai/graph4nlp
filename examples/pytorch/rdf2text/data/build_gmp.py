import pickle as pkl
import networkx as nx
import re

dataset = 'webnlg1/'
# dataset = 'webnlg2/'

train_input = open(dataset+'raw/train-webnlg-all-delex.triple').readlines()
train_output = open(dataset+'raw/train-webnlg-all-delex.lex').readlines()
train_rplc_list = pkl.load(open(dataset+'raw/train_rplc_list.pkl', 'rb'))

test_input = open(dataset+'raw/test-webnlg-all-delex.triple').readlines()
test_output = open(dataset+'raw/test-webnlg-all-delex.lex').readlines()
test_rplc_list = pkl.load(open(dataset+'raw/test_rplc_list.pkl', 'rb'))
# test_jump_idxs = open(dataset+'raw/test-jump-idxs.txt', 'w+')

dev_input = open(dataset+'raw/dev-webnlg-all-delex.triple').readlines()
dev_output = open(dataset+'raw/dev-webnlg-all-delex.lex').readlines()
dev_rplc_list = pkl.load(open(dataset+'raw/dev_rplc_list.pkl', 'rb'))

def write_gmp(dinput, doutput, rplc_list, part):
    graph_seqs = []
    jumps = []
    cnt=0
    new_triples = []
    new_texts = []
    idxs = []
    jump_idxs = []
    rplcs = []
    cnt_repeat_so = 0
    for oidx, (example, text, rplc) in enumerate(zip(dinput, doutput, rplc_list)):
        text = text.strip()
        # example = example.strip().lower()
        example = example.strip()
        g = nx.MultiDiGraph()
        # g = nx.DiGraph()
        ent_dict = {}

        old_example = example
        flag_repeat_so = False

        for triple in example.strip().split(' < TSP > '):
        # for triple in example.strip().split(' < tsp > '):
            s, p, o = triple.split(' | ')
            if s.split()[0] not in ent_dict.keys():
                ent_dict[s.split()[0]] = []
            if o.split()[0] not in ent_dict.keys():
                ent_dict[o.split()[0]] = []

            if ' '.join(s.split()) not in ent_dict[s.split()[0]]:
                ent_dict[s.split()[0]].append(' '.join(s.split()))
            # ent_dict[s.split()[0]] = list(set(ent_dict[s.split()[0]]))
            if ' '.join(o.split()) not in ent_dict[o.split()[0]]:
                ent_dict[o.split()[0]].append(' '.join(o.split()))
            # ent_dict[o.split()[0]] = list(set(ent_dict[o.split()[0]]))

        for k,v in ent_dict.items():
            if len(v)==1:
                continue

            v.sort()

            for x in v[1:]:
                example = example.replace(x, v[0])

        example_list = example.strip().split(' < TSP > ')
        # example_list = example.strip().split(' < tsp > ')

        if example == 'ENTITIES_1 MEANOFTRANSPORTATION | relatedMeanOfTransportation | DeSoto Custom < TSP > DeSoto Custom | relatedMeanOfTransportation | ENTITIES_2 MANUFACTURER < TSP > ENTITIES_1 MEANOFTRANSPORTATION | manufacturer | ENTITIES_2 MANUFACTURER':
            a = 0

        for triple in example_list:
            s, p, o = triple.split(' | ')
            if s==o:
                cnt_repeat_so += 1
                flag_repeat_so = True
                continue
            if g.get_edge_data(s, o) is not None and \
                    p in [v['edge_label'] for k, v in g.get_edge_data(s, o).items()]:
                continue
            g.add_edge(s, o, edge_label=p)

        indegree = nx.in_degree_centrality(g)
        outdegree = nx.out_degree_centrality(g)
        sources = []
        ends = []

        for k, v in indegree.items():
            if v == 0.0:
                sources.append(k)

        for k, v in outdegree.items():
            if v == 0.0:
                ends.append(k)

        if sources==[] or ends==[]:
            # print(example)
            cnt+=1
            jump_idxs.append(str(oidx))

            graph_seqs.append([ex.replace(' | ', ' ') for ex in example_list])
            jumps.append([])
            new_texts.append(text)
            new_triples.append(example)
            idxs.append(oidx)
            rplcs.append(rplc)

            continue

            # sources.append(sorted(indegree)[0])
            # ends.append(list(set(sorted(outdegree))-set(sources))[0])


        graph_seq = []
        jump = []
        # for source in sources:
        #     paths = nx.shortest_path(g, source=source)
        #     for end_node, path in paths.items():
        #         if len(path)==1 or end_node not in ends:
        #             continue
        #         graph_seq.append(path[0])
        #         for idx, node in enumerate(path):
        #             if idx==len(path)-1:
        #                 break
        #
        #             edge_data = g.get_edge_data(node, path[idx+1])
        #             for ed_k, ed_v in edge_data.items():
        #                 graph_seq.append(ed_v['edge_label'])
        #                 graph_seq.append(path[idx + 1])
        #                 if ed_k < len(edge_data)-1:
        #                     graph_seq.append(path[0])
        #
        #             if len(edge_data)!=1:
        #                 a = 0
        #             # graph_seq.append(g.get_edge_data(node, path[idx+1])[0]['edge_label'])
        #             # graph_seq.append(g.get_edge_data(node, path[idx+1])['edge_label'])
        #             # graph_seq.append(path[idx+1])
        #         jump.append(len(' '.join(graph_seq).split()))

        for source in sources:
            for end in ends:
                paths =list(nx.all_simple_paths(g, source=source, target=end))
                for path in paths:
                    graph_seq.append(path[0])
                    for idx, node in enumerate(path):
                        if idx==len(path)-1:
                            break

                        edge_data = g.get_edge_data(node, path[idx+1])
                        for ed_k, ed_v in edge_data.items():
                            graph_seq.append(ed_v['edge_label'])
                            graph_seq.append(path[idx + 1])
                            if ed_k < len(edge_data)-1:
                                graph_seq.append(path[0])

                    jump.append(len(' '.join(graph_seq).split()))

        if len(set(' '.join(graph_seq).split())) != len(set(example.replace(' | ', ' ').replace(' < TSP > ', ' ').split())) and flag_repeat_so==False:
            a = 0
        graph_seqs.append(graph_seq)
        jumps.append(jump)
        new_texts.append(text)
        new_triples.append(example)
        idxs.append(oidx)
        rplcs.append(rplc)

    all_list = []
    max_enc_len = 0
    for idx, tr, x, j, t, rplc in zip(idxs, new_triples, graph_seqs, jumps, new_texts, rplcs):
        dct = {}
        # dct['triples'] = tr.lower()
        dct['triples'] = tr
        dct['gmp_seqs'] = ' '.join(x)
        dct['gmp_jumps'] = j
        # dct['text'] = t.lower()
        dct['text'] = t
        dct['id'] = idx
        dct['rplc'] = rplc
        if j!=[] and max(j)>max_enc_len:
            max_enc_len = max(j)
        all_list.append(dct)

    print(max_enc_len)
    print(cnt)
    
    with open(dataset+'raw/{}_gmp_data.pt'.format(part),'wb+') as f:
        pkl.dump(all_list, f)

    # if part=='test':
    #     test_jump_idxs.write(' '.join(jump_idxs))

    return all_list

gmp_train = write_gmp(train_input, train_output, train_rplc_list, 'train')
gmp_test = write_gmp(test_input, test_output, test_rplc_list, 'test')
gmp_dev = write_gmp(dev_input, dev_output,dev_rplc_list, 'dev')