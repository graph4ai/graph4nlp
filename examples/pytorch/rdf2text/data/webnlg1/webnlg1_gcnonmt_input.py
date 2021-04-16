import networkx as nx
import re
import sys
import getopt
import pickle as pkl
import json

from collections import defaultdict
from examples.pytorch.rdf2text.data.benchmark_reader import Benchmark
from examples.pytorch.rdf2text.data.webnlg_baseline_input import select_files, relexicalise
# from examples.pytorch.rdf2text.data.webnlg2.webnlg_baseline_input import delexicalisation, select_files, relexicalise
#import webnlg_baseline_input as BASE_PREPROCESS
from unidecode import unidecode

from nltk import word_tokenize

import os

UNSEEN_CATEGORIES = ['NO_UNSEEN']
SEEN_CATEGORIES = ['Astronaut', 'Building', 'Monument', 'University', 'SportsTeam',
                   'WrittenWork', 'Food', 'ComicsCharacter', 'Airport', 'City']


if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")

def delexicalisation(out_src, out_trg, category, properties_objects):
    """
    Perform delexicalisation.
    :param out_src: source string
    :param out_trg: target string
    :param category: DBPedia category
    :param properties_objects: dictionary mapping properties to objects
    :return: delexicalised strings of the source and target; dictionary containing mappings of the replacements made
    """
    with open(os.path.dirname(os.path.realpath(sys.argv[0])) + '/raw/delex_dict.json') as data_file:
        data = json.load(data_file)

    # replace all occurrences of Alan_Bean to ASTRONAUT in input
    delex_subj = data[category]
    delex_src = out_src
    delex_trg = out_trg
    # for each instance, we save the mappings between nondelex and delex
    replcments = {}

    ent_cnt = 1

    # if 'veinticinco' in out_src:
    #     a = 0

    out_src_list = [x.strip() for x in out_src.strip().replace('< TSP >', '|').split(' | ')]
    for subject in delex_subj:
        subject = unidecode(subject.lower())
        clean_subj = ' '.join(re.split('(\W)', subject.replace('_', ' ')))
        clean_subj = re.sub('\s+', ' ', clean_subj.strip())

        if clean_subj in delex_src and clean_subj in out_src_list:
            replcments['ENTITIES_' + str(ent_cnt)] = ' '.join(clean_subj.split())  # remove redundant spaces

            # delex_src = delex_src.replace(clean_subj, 'ENTITIES_' + str(ent_cnt) + ' ' + category.upper())
            delex_src = re.sub('^{} \|'.format(clean_subj), 'ENTITIES_' + str(ent_cnt) + ' ' + category.upper()+' |', delex_src)
            delex_src = re.sub('\| {}$'.format(clean_subj), '| ENTITIES_' + str(ent_cnt) + ' ' + category.upper(), delex_src)
            delex_src = re.sub('\| {} <'.format(clean_subj), '| ENTITIES_' + str(ent_cnt) + ' ' + category.upper()+' <', delex_src)
            if re.match(clean_subj, delex_src) is None:
                delex_src = delex_src.replace(clean_subj, 'ENTITIES_' + str(ent_cnt) + ' ' + category.upper())

            # delex_trg = delex_trg.replace(clean_subj, 'ENTITIES_' + str(ent_cnt))
            delex_trg = re.sub(' ' + clean_subj + '[ ]*|^' + clean_subj + '[ ]*', ' ENTITIES_' + str(ent_cnt) + ' ',
                               delex_trg)

            clean_subj0 = clean_subj.replace(',', 'in')
            delex_trg = re.sub(' ' + clean_subj0 + '[ ]*|^' + clean_subj0 + '[ ]*', ' ENTITIES_' + str(ent_cnt) + ' ',
                               delex_trg)
            # delex_trg = delex_trg.replace(clean_subj0, 'ENTITIES_' + str(ent_cnt))

            clean_subj1 = clean_subj.replace(' - ', ' ')
            delex_trg = re.sub(' ' + clean_subj1 + '[ ]*|^' + clean_subj1 + '[ ]*', ' ENTITIES_' + str(ent_cnt) + ' ',
                               delex_trg)
            # delex_trg = delex_trg.replace(clean_subj1, 'ENTITIES_' + str(ent_cnt))

            clean_subj2 = clean_subj.replace(' , ', '')
            delex_trg = re.sub(' ' + clean_subj2 + '[ ]*|^' + clean_subj2 + '[ ]*', ' ENTITIES_' + str(ent_cnt) + ' ',
                               delex_trg)
            # delex_trg = delex_trg.replace(clean_subj2, 'ENTITIES_' + str(ent_cnt))

            clean_subj3 = clean_subj.replace(' , ', ' ')
            delex_trg = re.sub(' ' + clean_subj3 + '[ ]*|^' + clean_subj3 + '[ ]*', ' ENTITIES_' + str(ent_cnt) + ' ',
                               delex_trg)
            # delex_trg = delex_trg.replace(clean_subj3, 'ENTITIES_' + str(ent_cnt))

            clean_subj4 = clean_subj.replace(' , ', ' \( ') + ' \) '
            delex_trg = re.sub(' ' + clean_subj4 + '[ ]*|^' + clean_subj4 + '[ ]*', ' ENTITIES_' + str(ent_cnt) + ' ',
                               delex_trg)
            # delex_trg = delex_trg.replace(clean_subj4, 'ENTITIES_' + str(ent_cnt))

            for clean_subj5 in clean_subj.split(' , '):
                delex_trg = delex_trg.replace(clean_subj5, 'ENTITIES_' + str(ent_cnt))
                break

            ent_cnt += 1

    delex_src_list = [x.strip() for x in delex_src.strip().replace('< TSP >', '|').split(' | ')]
    # replace all occurrences of objects by PROPERTY in input
    for pro, obj in sorted(properties_objects.items()):
        clean_obj = ' '.join(re.split('(\W)', obj.replace('_', ' ').replace('"', '')))
        clean_obj = re.sub('\s+', ' ', clean_obj.strip())
        if clean_obj in delex_src and clean_obj.strip() in delex_src_list:
            replcments['ENTITIES_' + str(ent_cnt)] = ' '.join(clean_obj.split())  # remove redundant spaces

            # delex_src = re.sub(' ' + clean_obj + '[ ]*|^' + clean_obj + '[ ]*',
            #                    ' ENTITIES_' + str(ent_cnt) + ' ' + pro.upper() + ' ', delex_src)

            delex_src = re.sub('^{} \|'.format(clean_obj), 'ENTITIES_' + str(ent_cnt) + ' ' + pro.upper() + ' |',
                               delex_src)
            delex_src = re.sub('\| {}$'.format(clean_obj), '| ENTITIES_' + str(ent_cnt) + ' ' + pro.upper(),
                               delex_src)
            delex_src = re.sub('\| {} <'.format(clean_obj),
                               '| ENTITIES_' + str(ent_cnt) + ' ' + pro.upper() + ' <', delex_src)

            delex_trg = re.sub(' ' + clean_obj + '[ ]*|^' + clean_obj + '[ ]*', ' ENTITIES_' + str(ent_cnt) + ' ',
                               delex_trg)
            delex_trg = re.sub(' ' + clean_obj.lower() + '[ ]*|^' + clean_obj.lower() + '[ ]*',
                               ' ENTITIES_' + str(ent_cnt) + ' ', delex_trg)

            clean_obj = clean_obj.split('(')[0].strip()
            delex_src = re.sub(' ' + clean_obj + '[ ]*|^' + clean_obj + '[ ]*',
                               ' ENTITIES_' + str(ent_cnt) + ' ' + pro.upper() + ' ', delex_src)
            delex_trg = re.sub(' ' + clean_obj + '[ ]*|^' + clean_obj + '[ ]*',
                               ' ENTITIES_' + str(ent_cnt) + ' ', delex_trg)

            clean_obj0 = clean_obj.replace(',', 'in')
            delex_trg = re.sub(' ' + clean_obj0 + '[ ]*|^' + clean_obj0 + '[ ]*',
                               ' ENTITIES_' + str(ent_cnt) + ' ', delex_trg)
            # delex_trg = delex_trg.replace(clean_obj0, 'ENTITIES_' + str(ent_cnt))

            clean_obj1 = clean_obj.replace(' - ', ' ')
            delex_trg = re.sub(' ' + clean_obj1 + '[ ]*|^' + clean_obj1 + '[ ]*',
                               ' ENTITIES_' + str(ent_cnt) + ' ', delex_trg)
            # delex_trg = delex_trg.replace(clean_obj1, 'ENTITIES_' + str(ent_cnt))

            # clean_obj2 = clean_obj.replace(' - ', ' ')
            # delex_trg = re.sub(' ' + clean_obj2 + '[ ]*|^' + clean_obj2 + '[ ]*',
            #                    ' ENTITIES_' + str(ent_cnt) + ' ', delex_trg)
            # delex_trg = delex_trg.replace(clean_obj2, 'ENTITIES_' + str(ent_cnt))

            clean_obj3 = clean_obj.split('.')[0].strip()
            delex_trg = re.sub(' ' + clean_obj3 + '[ ]*|^' + clean_obj3 + '[ ]*',
                               ' ENTITIES_' + str(ent_cnt) + ' ', delex_trg)
            # delex_trg = delex_trg.replace(clean_obj3, 'ENTITIES_' + str(ent_cnt) + ' ')

            clean_obj4 = clean_obj.replace(',', 'and')
            delex_trg = re.sub(' ' + clean_obj4 + '[ ]*|^' + clean_obj4 + '[ ]*',
                               ' ENTITIES_' + str(ent_cnt) + ' ', delex_trg)
            # delex_trg = delex_trg.replace(clean_obj4, 'ENTITIES_' + str(ent_cnt))

            if '0 .' in clean_obj:
                clean_obj5 = clean_obj.split('0 .')[1].strip()
                clean_obj5 = '. ' + clean_obj5
                delex_trg = re.sub(' ' + clean_obj5 + '[ ]*|^' + clean_obj5 + '[ ]*',
                                   ' ENTITIES_' + str(ent_cnt) + ' ', delex_trg)
                # delex_trg = delex_trg.replace(clean_obj5, 'ENTITIES_' + str(ent_cnt) + ' ')

            # clean_obj6 = clean_obj.replace('u . s .', 'and')
            # delex_trg = delex_trg.replace(clean_obj6, 'ENTITIES_' + str(ent_cnt))

            if clean_obj in delex_trg.replace(' , ', ''):
                delex_trg = delex_trg.replace(' , ', '')
                delex_trg = re.sub(' ' + clean_obj + '[ ]*|^' + clean_obj + '[ ]*',
                                   ' ENTITIES_' + str(ent_cnt) + ' ', delex_trg)

            delex_trg = re.sub('\s+', ' ', delex_trg.strip())
            # if clean_obj in delex_trg.replace(' \' s ', ' '):
            #     delex_trg = delex_trg.replace(' \' s ', ' ').replace(clean_obj, 'ENTITIES_' + str(ent_cnt))

            ent_cnt += 1

    return delex_src.strip(), delex_trg.strip(), replcments

def buildGraph(srgGraph):  #, uniqueRelation=False):

    DG = nx.MultiDiGraph()
    for t in srgGraph.split("< TSP >"):
        t = t.strip().split(" | ")
        DG.add_edge(t[0],t[2], label=t[1]) # edge label is the property

    srcNodes = []
    srcEdgesLabels = []
    srcEdgesNode1 = []
    srcEdgesNode2 = []

    for eTriple in DG.edges(data='label'):
        rel = "_".join([x.strip() for x in eTriple[2].split()]) #eTriple[2].replace(" ", "_")
        subj = "_".join([x.strip() for x in eTriple[0].split()]) #eTriple[0].replace(" ", "_")
        obj = "_".join([x.strip() for x in eTriple[1].split()]) #eTriple[1].replace(" ", "_")

        relIdx = -1
        if not subj in srcNodes:
            srcNodes.append(subj)
        #if (uniqueRelation and not rel in srcNodes) or not uniqueRelation:
        srcNodes.append(rel)
        relIdx = len(srcNodes) - 1
        if not obj in srcNodes:
            srcNodes.append(obj)

        #srcEdges.append("|".join(["A0", str(srcNodes.index(subj)), str(srcNodes.index(rel))]))
        srcEdgesLabels.append("A0")
        srcEdgesNode1.append(str(srcNodes.index(subj)))
        srcEdgesNode2.append(str(relIdx))
        #srcEdges.append("|".join(["A1", str(srcNodes.index(obj)), str(srcNodes.index(rel))]))
        srcEdgesLabels.append("A1")
        srcEdgesNode1.append(str(srcNodes.index(obj)))
        srcEdgesNode2.append(str(relIdx))

    return " ".join(srcNodes), (" ".join(srcEdgesLabels), " ".join(srcEdgesNode1), " ".join(srcEdgesNode2))


def buildGraphWithNE(srgGraph):  #, uniqueRelation=False):
    # <class 'tuple'>: ('john', 'club', 'jong', 'van', 'den', 'brom', 'ajax')
    # if "wheeler" in srgGraph.lower():
    #     a=0
    DG = nx.MultiDiGraph()
    for t in srgGraph.split("< TSP >"):
        # t = t.strip().split("|")
        t = t.strip().split(" | ")
        try:
            DG.add_edge(t[0],t[2], label=t[1]) # edge label is the property
        except:
            a = 0

    srcNodes = []
    srcEdgesLabels = []
    srcEdgesNode1 = []
    srcEdgesNode2 = []

    for eTriple in DG.edges(data='label'):
        rel = "_".join([x.strip() for x in eTriple[2].split()])
        subj = [x.strip() for x in eTriple[0].split()]
        obj = [x.strip() for x in eTriple[1].split()]

        subjNodeDescendants = []
        objNodeDescendants = []
        subjNode = subj[0]
        if len(subj) > 1:
            subjNodeDescendants = subj[1:]
        objNode = obj[0]
        if len(obj):
            objNodeDescendants = obj[1:]
        if not subjNode in srcNodes:
            srcNodes.append(subjNode)
        srcNodes.append(rel)
        relIdx = len(srcNodes) - 1
        if not objNode in srcNodes:
            srcNodes.append(objNode)

        #srcEdges.append("|".join(["A0", str(srcNodes.index(subj)), str(srcNodes.index(rel))]))
        srcEdgesLabels.append("A0")
        srcEdgesNode1.append(str(srcNodes.index(subjNode)))
        srcEdgesNode2.append(str(relIdx))
        #srcEdges.append("|".join(["A1", str(srcNodes.index(obj)), str(srcNodes.index(rel))]))
        srcEdgesLabels.append("A1")
        srcEdgesNode1.append(str(srcNodes.index(objNode)))
        srcEdgesNode2.append(str(relIdx))

        if subjNodeDescendants:
            for neNode in subjNodeDescendants:
                srcNodes.append(neNode)
                nodeIdx = len(srcNodes) -1
                srcEdgesLabels.append("NE")
                srcEdgesNode1.append(str(nodeIdx))
                srcEdgesNode2.append(str(srcNodes.index(subjNode)))

        if objNodeDescendants:
            for neNode in objNodeDescendants:
                srcNodes.append(neNode)
                nodeIdx = len(srcNodes) -1
                srcEdgesLabels.append("NE")
                srcEdgesNode1.append(str(nodeIdx))
                srcEdgesNode2.append(str(srcNodes.index(objNode)))



    return " ".join(srcNodes), (" ".join(srcEdgesLabels), " ".join(srcEdgesNode1), " ".join(srcEdgesNode2))

def create_source_target(b, options, dataset, delex=True, relex=False, doCategory=[], negraph=False, lowercased=True, path=None):
    """
    Write target and source files, and reference files for BLEU.
    :param b: instance of Benchmark class
    :param options: string "delex" or "notdelex" to label files
    :param dataset: dataset part: train, dev, test
    :param delex: boolean; perform delexicalisation or not
    TODO:update parapms
    :return: if delex True, return list of replacement dictionaries for each example
    """

    source_out = []
    source_nodes_out = []
    source_edges_out_labels = []
    source_edges_out_node1 = []
    source_edges_out_node2 = []
    target_out = []
    rplc_list = []  # store the dict of replacements for each example
    for entr in b.entries:
        tripleset = entr.modifiedtripleset
        lexics = entr.lexs
        category = entr.category
        if doCategory and not category in doCategory:
            continue
        for lex in lexics:
            triples = ''
            properties_objects = {}
            tripleSep = ""
            triples_list = []

            for triple in tripleset.triples:
                triple.s = unidecode(triple.s)
                triple.o = unidecode(triple.o)

                if lowercased:
                    triple.s = triple.s.lower().replace('_', ' ').replace('"', '')
                    triple.o = triple.o.lower().replace('_', ' ').replace('"', '')

                    # if triple.s.lower() not in properties_objects.keys():
                    #     if triple.s in webnlg1_ent2type_new.keys():
                    #         # properties_objects[webnlg1_ent2type_new[triple.s]] = triple.s.lower()
                    #         properties_objects[triple.s.lower()] = 'ENTITIES_{} '.format(len(properties_objects))+webnlg1_ent2type_new[triple.s]
                    #     else:
                    #         properties_objects[triple.s.lower()] = 'ENTITIES_{} UNKNOWN'.format(len(properties_objects))

                    # if triple.o.lower() not in properties_objects.keys():
                    #     if triple.o in webnlg1_ent2type_new.keys():
                    #         # properties_objects[webnlg1_ent2type_new[triple.o]] = triple.o.lower()
                    #         properties_objects[triple.o.lower()] = 'ENTITIES_{} '.format(len(properties_objects))+webnlg1_ent2type_new[triple.o]
                    #     else:
                    #         # properties_objects[triple.p.upper()] = triple.o.lower()
                    #         properties_objects[triple.o.lower()] = 'ENTITIES_{} '.format(len(properties_objects))+triple.p.upper()

                # if delex:
                #     triples_list.append([properties_objects[triple.s],
                #                          triple.p,
                #                          properties_objects[triple.o]])
                #     triples += tripleSep + properties_objects[triple.s] \
                #                + '|' + triple.p + '|' + properties_objects[triple.o] + ' '
                #     tripleSep = "<TSP>"
                # else:
                triples_list.append([triple.s, triple.p, triple.o])
                triples += tripleSep + triple.s + '|' + triple.p + '|' + triple.o + ' '
                tripleSep = "<TSP>"

                properties_objects[triple.p] = triple.o

            triples = triples.replace('_', ' ').replace('"', '')

            out_src = re.sub('\s+', ' ', ' '.join(re.split('(\W)', triples)).strip())
            out_trg = re.sub('\s+', ' ', ' '.join(re.split('(\W)', lex.lex)).strip())

            # out_src = re.sub('\s+', ' ', triples)
            # out_trg = re.sub('\s+', ' ', lex.lex.strip()).replace('(','').replace(')','')

            out_src = unidecode(out_src)
            out_trg = unidecode(out_trg)

            if lowercased:
                out_trg = out_trg.lower()

            properties_objects = {k: v for k, v in sorted(properties_objects.items(),
                                                          key=lambda item: len(item[0]),
                                                          reverse=True)}

            if delex:
                out_src, out_trg, rplc_dict = delexicalisation(out_src, out_trg, category, properties_objects)
                rplc_list.append(rplc_dict)

            if negraph:
                source_nodes, source_edges = buildGraphWithNE(out_src)
            else:
                source_nodes, source_edges = buildGraph(out_src)
            source_nodes_out.append(source_nodes)
            source_edges_out_labels.append(source_edges[0])
            source_edges_out_node1.append(source_edges[1])
            source_edges_out_node2.append(source_edges[2])
            source_out.append(' '.join(out_src.split()))
            target_out.append(' '.join(out_trg.split()))

    #TODO: we could add a '-src-features.txt' if we want to attach features to nodes
    # if not relex:
    #     if doCategory==UNSEEN_CATEGORIES:
    #         options = options+'-unseen'
    #     we do not need to re-generate GCN input files when doing relexicalisation.. check this works ok
        # with open(dataset + '-webnlg-' + options + '-src-nodes.txt', 'w+', encoding='utf8') as f:
        #     f.write('\n'.join(source_nodes_out).lower() if (lowercased and not delex) else '\n'.join(source_nodes_out))
        # with open(dataset + '-webnlg-' + options + '-src-labels.txt', 'w+', encoding='utf8') as f:
        #     f.write('\n'.join(source_edges_out_labels))
        # with open(dataset + '-webnlg-' + options + '-src-node1.txt', 'w+', encoding='utf8') as f:
        #     f.write('\n'.join(source_edges_out_node1))
        # with open(dataset + '-webnlg-' + options + '-src-node2.txt', 'w+', encoding='utf8') as f:
        #     f.write('\n'.join(source_edges_out_node2))
        # with open(dataset + '-webnlg-' + options + '-tgt.txt', 'w+', encoding='utf8') as f:
        #     f.write('\n'.join(target_out).lower() if (lowercased and not delex) else '\n'.join(target_out))


    with open(path + dataset + '-webnlg-' + options + '.triple', 'w+', encoding='utf8') as f:
        f.write('\n'.join(source_out))
    with open(path + dataset + '-webnlg-' + options + '.lex', 'w+', encoding='utf8') as f:
        f.write('\n'.join(target_out).lower() if (lowercased and not delex)  else '\n'.join(target_out))

    # create separate files with references for multi-bleu.pl for dev set
    scr_refs = defaultdict(list)
    if (dataset == 'dev' or dataset.startswith('test')) and not delex:
        ##TODO: I think that taking only the nodes part is enough for BLEU scripts, see if we really nead the whole graph here in the src part
        for src, trg in zip(source_out, target_out):
            scr_refs[src].append(trg)
        # length of the value with max elements
        max_refs = sorted(scr_refs.values(), key=len)[-1]
        # keys = [key for (key, value) in sorted(scr_refs.items())]
        # values = [value for (key, value) in sorted(scr_refs.items())]
        keys = [key for (key, value) in scr_refs.items()]
        values = [value for (key, value) in scr_refs.items()]
        # write the source file not delex
        with open(path + dataset + "-" + options + '-source.triple', 'w+', encoding='utf8') as f:
            f.write('\n'.join(keys))
        # write references files
        for j in range(0, len(max_refs)):
            with open(path + dataset + "-" + options + '-reference' + str(j) + '.lex', 'w+', encoding='utf8') as f:
                out = ''
                for ref in values:
                    try:
                        out += ref[j].lower()  + '\n' if (lowercased and not delex) else ref[j] + '\n'
                    except:
                        out += '\n'
                f.write(out)
                f.close()

        #write reference files for E2E evaluation metrics
        with open(path + dataset + "-" + options + '-conc.txt', 'w+', encoding='utf8') as f:
            for ref in values:
                for j in range(len(ref)):
                    f.write( ref[j].lower()  + '\n' if (lowercased and not delex) else ref[j] + '\n')
                f.write("\n")
            f.close()

    return rplc_list

# def input_files(path, filepath=None, relex=False, parts=['test'],
def input_files(path, filepath=None, relex=False, parts=['train', 'dev'],
                doCategory=[],
                negraph=True,
                lowercased=True,
                fileid=None):
    """
    Read the corpus, write train and dev files.
    :param path: directory with the WebNLG benchmark
    :param filepath: path to the prediction file with sentences (for relexicalisation)
    :param relex: boolean; do relexicalisation or not
    :param parts: partition to process
    :param negraph: whether to add edges for multi-word entitites
    :param lowercased: whether to do all lowercased for the notdelex version of the files
    :return:
    """

    rplc_list_dev_delex = None
    options = ['all-delex', 'all-notdelex']  # generate files with/without delexicalisation
    # options = ['all-notdelex']  # generate files with/without delexicalisation
    for part in parts:
        for option in options:
            if part.startswith('test'):
                files = select_files(path + part, size=0)
            else:
                files = select_files(path + part, size=(1, 8))
            if doCategory == UNSEEN_CATEGORIES:
                files = [files[1]]
            b = Benchmark()
            b.fill_benchmark(files)
            if option == 'all-delex':
                rplc_list = create_source_target(
                    b, option, part, delex=True, relex=relex, doCategory=doCategory, negraph=negraph, lowercased=lowercased, path=path)
                print('Total of {} files processed in {} with {} mode'.format(len(files), part, option))

                with open(path + part + '_rplc_list.pkl', 'wb+') as f:
                    pkl.dump(rplc_list, f)
            elif option == 'all-notdelex':
                rplc_list = create_source_target(
                    b, option, part, delex=False, relex=relex, doCategory=doCategory, negraph=negraph, lowercased=lowercased, path=path)
                print('Total of {} files processed in {} with {} mode'.format(len(files), part, option))

                # with open(part + '_rplc_list.pkl', 'wb+') as f:
                #     pkl.dump(rplc_list, f)

            if (part == 'dev' or part.startswith('test')) and option == 'all-delex':
                rplc_list_dev_delex = rplc_list


    if relex and rplc_list_dev_delex:
        relexicalise(filepath, rplc_list_dev_delex, fileid, part, lowercased=lowercased)
    print('Files necessary for training/evaluating are written on disc.')


def main(argv):
    usage = 'usage:\npython3 webnlg_gcnonmt_input.py -i <data-directory> [-p PARTITION] [-c CATEGORIES] [-e NEGRAPH]' \
           '\ndata-directory is the directory where you unzipped the archive with data'\
           '\nPARTITION which partition to process, by default test/devel will be done.'\
           '\n-c is seen or unseen if we want to filter the test seen per category.' \
           '\n-l generate all source/target files in lowercase.'
    try:
        opts, args = getopt.getopt(argv, 'i:p:c:el', ['inputdir=','partition=', 'category=', 'negraph=', 'lowercased='])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    input_data = False
    ngraph = False
    partition = None
    category = None
    lowercased = False
    for opt, arg in opts:
        if opt in ('-i', '--inputdir'):
            inputdir = arg
            input_data = True
        elif opt in ('-p', '--partition'):
            partition = arg
        elif opt in ('-c', '--category'):
            category = arg
        elif opt in ('-e', '--negraph'):
            ngraph = True
        elif opt in ('-l', '--lowercased'):
            lowercased = True
        else:
            print(usage)
            sys.exit()
    if not input_data:
        print(usage)
        sys.exit(2)
    print('Input directory is {}, NE={}, lowercased={}'.format(inputdir, ngraph, lowercased))
    if partition:
        if category=='seen':
            input_files(inputdir, parts=[partition], doCategory=SEEN_CATEGORIES,
                        negraph=ngraph, lowercased=lowercased)
        elif category=='unseen':
           input_files(inputdir, parts=[partition], doCategory=UNSEEN_CATEGORIES)
        else:
            input_files(inputdir, parts=[partition], negraph=ngraph, lowercased=lowercased)
            #this does all in the fnput file, which is normally 'test'
    else:
        input_files(inputdir)


if __name__ == "__main__":
    main(sys.argv[1:])
