import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from dgl.data import register_data_args

from ...modules.graph_construction import NodeEmbeddingBasedGraphConstruction
from ...modules.utils.padding_utils import pad_2d_vals_no_size
from ...modules.utils.vocab_utils import VocabModel


def main(args, seed):
    # Configure
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not args.no_cuda and torch.cuda.is_available():
        print("[ Using CUDA ]")
        device = torch.device("cuda" if args.gpu < 0 else "cuda:%d" % args.gpu)
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")

    raw_text_data = [["I like nlp.", "Same here!"], ["I like graph.", "Same here!"]]

    vocab_model = VocabModel(
        raw_text_data, max_word_vocab_size=None, min_word_vocab_freq=1, word_emb_size=300
    )

    src_text_seq = list(zip(*raw_text_data))[0]

    # Test raw_text_to_init_graph method of dynamic graph construction class
    text_graphs = []
    for each in src_text_seq:
        tmp_graph = NodeEmbeddingBasedGraphConstruction.raw_text_to_init_graph(each)
        text_graphs.append(tmp_graph)

    src_idx_seq = [vocab_model.word_vocab.to_index_sequence(each) for each in src_text_seq]
    src_len = torch.LongTensor([len(each) for each in src_idx_seq]).to(device)
    num_seq = torch.LongTensor([len(src_len)]).to(device)
    input_tensor = torch.LongTensor(pad_2d_vals_no_size(src_idx_seq)).to(device)
    print("input_tensor: {}".format(input_tensor.shape))

    embedding_styles = {
        "word_emb_type": "w2v",
        "node_edge_emb_strategy": "bilstm",
        "seq_info_encode_strategy": "none",
    }

    gl = NodeEmbeddingBasedGraphConstruction(
        vocab_model.word_vocab,
        embedding_styles,
        input_size=args.gl_num_hidden,
        hidden_size=args.gl_num_hidden,
        top_k_neigh=args.gl_topk,
        device=device,
    )

    gl.to(device)
    print(gl)

    graph = gl(input_tensor, src_len, num_seq)
    print("\nadj:", graph.adjacency_matrix().to_dense())
    print("node feat: ", graph.ndata["node_feat"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DynamicGraphConstruction")
    register_data_args(parser)
    parser.add_argument("--no-cuda", action="store_true", default=False, help="use CPU")
    parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use.")
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--gl-num-hidden",
        type=int,
        default=16,
        help="number of hidden units for dynamic graph construction",
    )
    parser.add_argument(
        "--gl-topk", type=int, default=200, help="top k for dynamic graph construction"
    )
    parser.add_argument(
        "--gl-type",
        type=str,
        default="node_emb",
        help=r"dynamic graph construction algorithm type, \
                        {'node_emb', 'node_edge_emb' and 'node_emb_refined'},\
                        default: 'node_emb'",
    )
    parser.add_argument(
        "--init-adj-alpha",
        type=float,
        default=0.8,
        help="alpha ratio for combining initial graph adjacency matrix",
    )
    parser.add_argument("--num-hidden", type=int, default=16, help="number of hidden units")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=False,
        help="indicates whether to use early stop or not",
    )
    parser.add_argument("--patience", type=int, default=100, help="early stopping patience")
    parser.add_argument(
        "--fastmode", action="store_true", default=False, help="skip re-evaluate the validation set"
    )
    parser.add_argument(
        "--save-model-path", type=str, default="checkpoint", help="path to the best saved model"
    )
    args = parser.parse_args()
    args.save_model_path = "{}_{}_gl_type_{}_gl_topk_{}_init_adj_alpha_{}".format(
        args.save_model_path, args.dataset, args.gl_type, args.gl_topk, args.init_adj_alpha
    )
    print(args)
    main(args, 1234)
