import sys
import getopt
from examples.pytorch.rdf2text.data.webnlg1.webnlg1_gcnonmt_input import input_files, SEEN_CATEGORIES, UNSEEN_CATEGORIES


def relex_main(argv):
    usage = 'usage:\npython3 webnlg_gcnonmt_relexicalise.py -i <data-directory> -f <prediction-file> [-p PARTITION] [-c CATEGORIES] [-g GPUID]' \
           '\ndata-directory is the directory where you unzipped the archive with data' \
            '\nprediction-file is the path to the generated file baseline_predictions.txt' \
            ' (e.g. documents/baseline_predictions.txt)'\
           '\n PARTITION which partition to process, by default test/devel will be done.'\
           '\n-c is seen or unseen if we want to filter the test seen per category.' \
            '\n-g the gpuid is given to number the output files differently for each gpu task running.' \
            '\n-l the lowercase relexicalised output .'
    try:
        opts, args = getopt.getopt(argv, 'i:f:p:c:g:l', ['inputdir=', 'filedir=', 'partition=', 'category=', 'gpuid=', 'lowercased='])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    input_data = False
    input_filepath = False
    partition = None
    category = None
    gpuid = None
    lowercased = False
    for opt, arg in opts:
        if opt in ('-i', '--inputdir'):
            inputdir = arg
            input_data = True
        elif opt in ('-f', '--filedir'):
            filepath = arg
            input_filepath = True
        elif opt in ('-p', '--partition'):
            partition = arg
        elif opt in ('-c', '--category'):
            category = arg
        elif opt in ('-g', '--gpuid'):
            gpuid = arg
        elif opt in ('-l', '--lowercased'):
            lowercased = True
        else:
            print(usage)
            sys.exit()
    if not input_data or not input_filepath:
        print(usage)
        sys.exit(2)
    # print('Input directory is', inputdir)
    # print('Path to the file is', filepath)
    # print('Input directory is ', inputdir)
    if partition:
        if category=='seen':
            input_files(inputdir, filepath, relex=True, parts=[partition], doCategory=SEEN_CATEGORIES, fileid=gpuid, lowercased=lowercased)
        #elif category=='unseen':
        #    input_files(inputdir, filepath, relex=True, parts=[partition], doCategory=UNSEEN_CATEGORIES)
        else:
            input_files(inputdir, filepath, relex=True, parts=[partition], fileid=gpuid, lowercased=lowercased)
    else:
        input_files(inputdir, filepath, relex=True)

if __name__ == "__main__":
    relex_main(sys.argv[1:])

'''
-i
../data/webnlg2/
-f
../data/translate_result/webnlg2/webnlg2_gtr2_copy_43_translate_relex.txt
-p
test
-c
seen
-l
'''

'''
-i
raw/
-f
/raid/user8/graph4nlp-rdf2text/out/webnlg2/gcn_ckpt/pred.txt
-p
test
-c
seen
-l
'''