import argparse
import copy
import pickle as pkl
import random
import warnings

import numpy as np
import torch

from .train import do_generate
from .utils import data_utils, graph_utils
from .utils.tree import Tree

warnings.filterwarnings('ignore')

def define_queries_order(str1, str2):
    c_str = str2.replace(") ,", ") @")
    c_list = c_str.strip().split("@")
    flag = True
    r_str = str1
    for c_phrase in c_list:
        if c_phrase.strip() in str1:
            r_str = r_str.replace(c_phrase.strip(), "")
        else:
            flag = False
    if len(r_str) != (r_str.count(",") + r_str.count(" ")):
        flag = False
    return flag

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument(
        '-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument(
        '-display', type=int, default=1, help='whether display on console')
    main_arg_parser.add_argument('-data_dir', type=str,
                                 default='../data/GraphConstruction', help='graph and tree data_dir')
    main_arg_parser.add_argument(
        '-model', type=str, default='../experiments/best_output_model', help='best model output')
    main_arg_parser.add_argument(
        '-model_dir', type=str, default='../experiments/', help='models saved')
    main_arg_parser.add_argument(
        '-num_batch', type=int, default=25)
    main_arg_parser.add_argument(
        '-seed', type=int, default=123, help='torch manual random number generator seed')
    main_arg_parser.add_argument(
        '-mode', type=int, default=1)
    main_arg_parser.add_argument('-para_for_sibling', type=float, default=0.001)


    args = main_arg_parser.parse_args()

    managers = pkl.load(open("{}/map.pkl".format(args.data_dir), "rb"))
    word_manager, form_manager = managers

    # print form_manager.idx2symbol
    if args.mode == 1:
        data = pkl.load(open("{}/valid.pkl".format(args.data_dir), "rb"))
        graph_test_list = graph_utils.read_graph_data(
            "{}/graph.valid".format(args.data_dir))
    else:
        data = pkl.load(open("{}/test.pkl".format(args.data_dir), "rb"))
        graph_test_list = graph_utils.read_graph_data(
            "{}/graph.test".format(args.data_dir))

    max_acc = 0
    max_index = 0

    model_num = args.num_batch - 1
    while(1):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print(model_num)
        try:
            checkpoint = torch.load(
                args.model_dir + "model_g2t" + str(model_num))
        except BaseException:
            break

        encoder = checkpoint["encoder"]
        decoder = checkpoint["decoder"]
        attention_decoder = checkpoint["attention_decoder"]

        encoder.eval()
        decoder.eval()
        attention_decoder.eval()

        reference_list = []
        candidate_list = []

        add_acc = 0.0
        for i in range(len(data)):
            x = data[i]
            reference = x[1]
            graph_batch = graph_utils.cons_batch_graph([graph_test_list[i]])
            graph_input = graph_utils.vectorize_batch_graph(
                graph_batch, word_manager)
            candidate = do_generate(encoder, decoder, attention_decoder, graph_input,
                                    word_manager, form_manager, args, -1, checkpoint)
            candidate = [int(c) for c in candidate]
            num_left_paren = sum(
                1 for c in candidate if form_manager.idx2symbol[int(c)] == "(")
            num_right_paren = sum(
                1 for c in candidate if form_manager.idx2symbol[int(c)] == ")")
            diff = num_left_paren - num_right_paren
            if diff > 0:
                for i in range(diff):
                    candidate.append(form_manager.symbol2idx[")"])
            elif diff < 0:
                candidate = candidate[:diff]
            ref_str = data_utils.convert_to_string(reference, form_manager)
            cand_str = data_utils.convert_to_string(candidate, form_manager)
            reference_list.append(reference)
            candidate_list.append(candidate)
            # if ref_str != cand_str and define_queries_order(ref_str, cand_str) == True:
            #     add_acc += 1.0
        val_acc = data_utils.compute_tree_accuracy(
            candidate_list, reference_list, form_manager) + add_acc/len(data)
        print(("ACCURACY = {}\n".format(val_acc)))
        if val_acc >= max_acc:
            max_acc = val_acc
            max_index = model_num

        model_num += args.num_batch
    print("max accuracy:", max_acc)
    best_valid_model = torch.load(
        args.model_dir + "model_g2t" + str(max_index))
    torch.save(best_valid_model, args.model)