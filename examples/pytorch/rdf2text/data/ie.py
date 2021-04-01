import argparse
import stanfordcorenlp
from graph4nlp.pytorch.modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
import warnings

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-generated_file', '--generated_file', required=True, type=str, help='path to the generated file')
    parser.add_argument('-port', '--port', type=int, default=9000, help='path to the config file')
    # parser.add_argument('--grid_search', action='store_true', help='flag: grid search')
    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':
    cfg = get_args()
    port = cfg['port']
    data_items = open(cfg['generated_file'], 'r').readlines()
    topology_builder = IEBasedGraphConstruction

    print('Connecting to stanfordcorenlp server...')
    processor = stanfordcorenlp.StanfordCoreNLP('http://localhost', port=cfg['port'], timeout=300000)

    props_coref = {
        'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref',
        "tokenize.options":
            "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
        "tokenize.whitespace": False,
        'ssplit.isOneSentence': False,
        'outputFormat': 'json'
    }
    props_openie = {
        'annotators': 'tokenize, ssplit, pos, ner, parse, openie',
        "tokenize.options":
            "splitHyphenated=true,normalizeParentheses=true,normalizeOtherBrackets=true",
        "tokenize.whitespace": False,
        'ssplit.isOneSentence': False,
        'outputFormat': 'json',
        "openie.triple.strict": "true"
    }
    processor_args = [props_coref, props_openie]

    print('CoreNLP server connected.')
    pop_idxs = []
    ret = []
    graph_content_collect = []
    for cnt, item in enumerate(data_items):
        if cnt % 1000 == 0:
            print("Port {}, processing: {} / {}".format(port, cnt, len(data_items)))
        try:
            graph = topology_builder.topology(raw_text_data=item,
                                              nlp_processor=processor,
                                              processor_args=processor_args,
                                              merge_strategy=None,
                                              edge_strategy=None,
                                              verbase=False,
                                              ret_graph=False)
            # item.graph = graph
        except Exception as msg:
            pop_idxs.append(cnt)
            graph = None
            warnings.warn(RuntimeWarning(msg))
        ret.append(graph)
        graph_content_collect.append(graph['graph_content'])
    ret = [x for idx, x in enumerate(ret) if idx not in pop_idxs]
    print(len(pop_idxs))
    with open('ie_results.txt', 'w') as f:
        for x in graph_content_collect:
            f.write(str(x)+'\n')