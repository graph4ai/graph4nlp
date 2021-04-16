import os
import random
import re
import json
import sys
import getopt
from operator import itemgetter
from collections import defaultdict
# from benchmark_reader import Benchmark
from unidecode import unidecode

def select_files(topdir, category='', size=(1, 8)):
    """
    Collect all xml files from a benchmark directory.
    :param topdir: directory with benchmark
    :param category: specify DBPedia category to retrieve texts for a specific category (default: retrieve all)
    :param size: specify size to retrieve texts of specific size (default: retrieve all)
    :return: list of tuples (full path, filename)
    """
    if size==0:
        finaldirs = [topdir]
    else:
        finaldirs = [topdir+'/'+str(item)+'triples' for item in range(size[0], size[1])]

    finalfiles = []
    for item in finaldirs:
        finalfiles += [(item, filename) for filename in os.listdir(item)]
    if category:
        finalfiles = []
        for item in finaldirs:
            finalfiles += [(item, filename) for filename in os.listdir(item) if category in filename]
    return finalfiles

def delexicalisation(out_src, out_trg, category, properties_objects):
    """
    Perform delexicalisation.
    :param out_src: source string
    :param out_trg: target string
    :param category: DBPedia category
    :param properties_objects: dictionary mapping properties to objects
    :return: delexicalised strings of the source and target; dictionary containing mappings of the replacements made
    """
    try:
        with open(os.path.dirname(os.path.realpath(sys.argv[0]))+'/data/webnlg2/raw/delex_dict.json') as data_file:
            data = json.load(data_file)
    except:
        with open(os.path.dirname(os.path.realpath(sys.argv[0]))+'/raw/delex_dict.json') as data_file:
            data = json.load(data_file)

    # replace all occurrences of Alan_Bean to ASTRONAUT in input
    delex_subj = data[category]
    delex_src = out_src
    delex_trg = out_trg
    # for each instance, we save the mappings between nondelex and delex
    replcments = {}

    ent_cnt=1

    out_src_list = [x.strip() for x in out_src.strip().replace('< TSP >', '|').split(' | ')]
    for subject in delex_subj:
        subject = unidecode(subject.lower())
        clean_subj = ' '.join(re.split('(\W)', subject.replace('_', ' ')))
        clean_subj = re.sub('\s+', ' ', clean_subj.strip())

        if clean_subj in delex_src and clean_subj in out_src_list:
            replcments['ENTITIES_' + str(ent_cnt)] = ' '.join(clean_subj.split())  # remove redundant spaces

            delex_src = delex_src.replace(clean_subj, 'ENTITIES_'+ str(ent_cnt) + ' ' + category.upper())
            delex_trg = delex_trg.replace(clean_subj, 'ENTITIES_'+ str(ent_cnt))

            clean_subj0 = clean_subj.replace(',', 'in')
            delex_trg = delex_trg.replace(clean_subj0, 'ENTITIES_' + str(ent_cnt))

            clean_subj1 = clean_subj.replace(' - ', ' ')
            delex_trg = delex_trg.replace(clean_subj1, 'ENTITIES_' + str(ent_cnt))

            clean_subj2 = clean_subj.replace(' , ', '')
            delex_trg = delex_trg.replace(clean_subj2, 'ENTITIES_' + str(ent_cnt))

            clean_subj3 = clean_subj.replace(' , ', ' ')
            delex_trg = delex_trg.replace(clean_subj3, 'ENTITIES_' + str(ent_cnt))

            clean_subj4 = clean_subj.replace(' , ', ' ( ') + ' ) '
            delex_trg = delex_trg.replace(clean_subj4, 'ENTITIES_' + str(ent_cnt))

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

            delex_src = re.sub(' '+clean_obj+'[ ]*|^'+clean_obj+'[ ]*', ' ENTITIES_'+ str(ent_cnt) + ' ' + pro.upper()+' ', delex_src)
            delex_trg = re.sub(' '+clean_obj+'[ ]*|^'+clean_obj+'[ ]*', ' ENTITIES_'+ str(ent_cnt)+' ', delex_trg)
            delex_trg = re.sub(' '+clean_obj.lower()+'[ ]*|^'+clean_obj.lower()+'[ ]*',
                               ' ENTITIES_'+ str(ent_cnt)+' ', delex_trg)

            clean_obj = clean_obj.split('(')[0].strip()
            delex_src = re.sub(' ' + clean_obj + '[ ]*|^' + clean_obj + '[ ]*',
                               ' ENTITIES_' + str(ent_cnt) + ' ' + pro.upper()+' ', delex_src)
            delex_trg = re.sub(' '+clean_obj+'[ ]*|^'+clean_obj+'[ ]*',
                               ' ENTITIES_'+ str(ent_cnt)+' ', delex_trg)

            clean_obj0 = clean_obj.replace(',', 'in')
            delex_trg = delex_trg.replace(clean_obj0, 'ENTITIES_' + str(ent_cnt))

            clean_obj1 = clean_obj.replace(' - ', ' ')
            delex_trg = delex_trg.replace(clean_obj1, 'ENTITIES_' + str(ent_cnt))

            clean_obj2 = clean_obj.replace(' - ', ' ')
            delex_trg = delex_trg.replace(clean_obj2, 'ENTITIES_' + str(ent_cnt))

            clean_obj3 = clean_obj.split('.')[0].strip()
            delex_trg = delex_trg.replace(clean_obj3, 'ENTITIES_' + str(ent_cnt) +' ')

            clean_obj4 = clean_obj.replace(',', 'and')
            delex_trg = delex_trg.replace(clean_obj4, 'ENTITIES_' + str(ent_cnt))

            if '0 .' in clean_obj:
                clean_obj5 = clean_obj.split('0 .')[1].strip()
                clean_obj5 = '. '+clean_obj5
                delex_trg = delex_trg.replace(clean_obj5, 'ENTITIES_' + str(ent_cnt) + ' ')

            clean_obj5 = clean_obj.replace('u . s .', 'and')
            delex_trg = delex_trg.replace(clean_obj5, 'ENTITIES_' + str(ent_cnt))

            if clean_obj in delex_trg.replace(' , ', ''):
                delex_trg = delex_trg.replace(' , ', '').replace(clean_obj, 'ENTITIES_' + str(ent_cnt))

            delex_trg = re.sub('\s+', ' ', delex_trg.strip())
            # if clean_obj in delex_trg.replace(' \' s ', ' '):
            #     delex_trg = delex_trg.replace(' \' s ', ' ').replace(clean_obj, 'ENTITIES_' + str(ent_cnt))

            ent_cnt += 1

    return delex_src.strip(), delex_trg.strip(), replcments

def create_source_target(b, options, dataset, delex=True):
    """
    Write target and source files, and reference files for BLEU.
    :param b: instance of Benchmark class
    :param options: string "delex" or "notdelex" to label files
    :param dataset: dataset part: train, dev, test
    :param delex: boolean; perform delexicalisation or not
    :return: if delex True, return list of replacement dictionaries for each example
    """
    source_out = []
    target_out = []
    rplc_list = []  # store the dict of replacements for each example
    for entr in b.entries:
        tripleset = entr.modifiedtripleset
        # triples = ''
        triples = []
        properties_objects = {}
        for triple in tripleset.triples:
            # triples += triple.s + ' ' + triple.p + ' ' + triple.o + ' '
            triples.append(triple.s + ' ' + triple.p + ' ' + triple.o + ' ')
            properties_objects[triple.p] = triple.o
        #random.shuffle(triples)

        #print(triples)
        triples.reverse()
        #print(triples)

        triples = ' '.join(triples)
        triples = triples.replace('_', ' ').replace('"', '')
        lexics = entr.lexs
        category = entr.category
        out_src = ' '.join(re.split('(\W)', triples))
        for lex in lexics:
            # separate punct signs from text
            out_trg = ' '.join(re.split('(\W)', lex.lex))
            if delex:
                out_src, out_trg, rplc_dict = delexicalisation(out_src, out_trg, category, properties_objects)
                rplc_list.append(rplc_dict)
            # delete white spaces
            source_out.append(' '.join(out_src.split()))
            target_out.append(' '.join(out_trg.split()))

    # shuffle two lists in the same way
    random.seed(10)
    if delex:
        corpus = list(zip(source_out, target_out, rplc_list))
        random.shuffle(corpus)
        source_out, target_out, rplc_list = zip(*corpus)
    else:
        corpus = list(zip(source_out, target_out))
        random.shuffle(corpus)
        source_out, target_out = zip(*corpus)

    with open(dataset + '-webnlg-' + options + '.triple', 'w+', encoding='utf8') as f:
        f.write('\n'.join(source_out))
    with open(dataset + '-webnlg-' + options + '.lex', 'w+', encoding='utf8') as f:
        f.write('\n'.join(target_out))

    # create separate files with references for multi-bleu.pl for dev set
    scr_refs = defaultdict(list)
    if dataset == 'dev' and not delex:
        for src, trg in zip(source_out, target_out):
            scr_refs[src].append(trg)

        # length of the value with max elements
        max_refs = sorted(scr_refs.values(), key=len)[-1]
        keys = [key for (key, value) in sorted(scr_refs.items())]
        values = [value for (key, value) in sorted(scr_refs.items())]
        # write the source file not delex
        with open(options + '-source.triple', 'w+', encoding='utf8') as f:
            f.write('\n'.join(keys))
        # write references files
        for j in range(0, len(max_refs)):
            with open(options + '-reference' + str(j) + '.lex', 'w+', encoding='utf8') as f:
                out = ''
                for ref in values:
                    try:
                        out += ref[j] + '\n'
                    except:
                        out += '\n'
                f.write(out)

    return rplc_list


def relexicalise(predfile, rplc_list, fileid, part='dev', lowercased=True):
    """
    Take a file from seq2seq output and write a relexicalised version of it.
    :param rplc_list: list of dictionaries of replacements for each example (UPPER:not delex item)
    :return: list of predicted sentences
    """
    relex_predictions = []
    with open(predfile, 'r') as f:
        predictions = [line for line in f]
    if rplc_list:
        for i, pred in enumerate(predictions):
            # replace each item in the corresponding example
            try:
                rplc_dict = rplc_list[i]
            except:
                a=0
            relex_pred = pred

            for key in sorted(rplc_dict):
                relex_pred = relex_pred.replace(key + ' ', rplc_dict[key] + ' ')
            relex_predictions.append(relex_pred)
    else:
        relex_predictions = predictions

    # with open('relexicalised_predictions_full.txt', 'w+') as f:
        # f.write(''.join(relex_predictions))

    # create a mapping between not delex triples and relexicalised sents
    # with open('examples/pytorch/rdf2text/data/webnlg2/raw/'+part+'-webnlg-all-notdelex.triple', 'r') as f:
    with open('raw/'+part+'-webnlg-all-notdelex.triple', 'r') as f:
        dev_sources = [line.strip() for line in f]
    src_gens = {}
    for src, gen in zip(dev_sources, relex_predictions):
        src_gens[src] = gen  # need only one lex, because they are the same for a given triple

    # write generated sents to a file in the same order as triples are written in the source file
    # with open('examples/pytorch/rdf2text/data/webnlg2/raw/'+part+'-all-notdelex-source.triple', 'r') as f:
    with open('raw/'+part+'-all-notdelex-source.triple', 'r') as f:
        triples = [line.strip() for line in f]
    outfileName = predfile.split('/')[-1].split('.')[0]+ '_relexicalised_predictions.txt'
    if fileid:
        outfileName = 'relexicalised_predictions'+str(fileid)+'.txt'
    with open(outfileName, 'w+', encoding='utf8') as f:
        for triple in triples:
            relexoutput = src_gens[triple]
            if lowercased:
                relexoutput = relexoutput.lower()
            f.write(relexoutput)

    return relex_predictions



def input_files(path, filepath=None, relex=False):
    """
    Read the corpus, write train and dev files.
    :param path: directory with the WebNLG benchmark
    :param filepath: path to the prediction file with sentences (for relexicalisation)
    :param relex: boolean; do relexicalisation or not
    :return:
    """
    rplc_list_dev_delex = None
    parts = ['train', 'dev']
    options = ['all-delex', 'all-notdelex']  # generate files with/without delexicalisation
    for part in parts:
        for option in options:
            files = select_files(path + part, size=(1, 8))
            b = Benchmark()
            b.fill_benchmark(files)
            if option == 'all-delex':
                rplc_list = create_source_target(b, option, part, delex=True)
                print('Total of {} files processed in {} with {} mode'.format(len(files), part, option))
            elif option == 'all-notdelex':
                rplc_list = create_source_target(b, option, part, delex=False)
                print('Total of {} files processed in {} with {} mode'.format(len(files), part, option))
            if part == 'dev' and option == 'all-delex':
                rplc_list_dev_delex =rplc_list

    if relex and rplc_list_dev_delex:
        relexicalise(filepath, rplc_list_dev_delex)
    print('Files necessary for training/evaluating are written on disc.')


def main(argv):
    usage = 'usage:\npython3 webnlg_baseline_input.py -i <data-directory> [-s]' \
           '\ndata-directory is the directory where you unzipped the archive with data'
    try:
        opts, args = getopt.getopt(argv, 'i:s', ['inputdir=','shuffle'])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    input_data = False
    shuffleTripleSet = False
    for opt, arg in opts:
        if opt in ('-i', '--inputdir'):
            inputdir = arg
            input_data = True
        elif opt in ('-s', '--shuffle'):
            shuffleTripleSet = True
        else:
            print(usage)
            sys.exit()
    if not input_data:
        print(usage)
        sys.exit(2)
    print('Input directory is ', inputdir)
    input_files(inputdir)


if __name__ == "__main__":
    main(sys.argv[1:])
