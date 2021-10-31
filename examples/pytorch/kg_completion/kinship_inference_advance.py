import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from evaluation import ranking_and_hits
from model import Complex, ConvE, Distmult, GCNComplex, GCNDistMult, GGNNDistMult
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.utils.global_config import Config

np.set_printoptions(precision=3)

cudnn.benchmark = True


def main(args, model_path):
    input_keys = ["e1", "rel", "rel_eval", "e2", "e2_multi1", "e2_multi2"]
    p = Pipeline(args.data, keys=input_keys)
    p.load_vocabs()
    vocab = p.state["vocab"]

    test_rank_batcher = StreamBatcher(
        args.data,
        "test_ranking",
        args.test_batch_size,
        randomize=False,
        loader_threads=args.loader_threads,
        keys=input_keys,
    )

    if args.model is None:
        model = ConvE(args, vocab["e1"].num_token, vocab["rel"].num_token)
    elif args.model == "conve":
        model = ConvE(args, vocab["e1"].num_token, vocab["rel"].num_token)
    elif args.model == "distmult":
        model = Distmult(args, vocab["e1"].num_token, vocab["rel"].num_token)
    elif args.model == "complex":
        model = Complex(args, vocab["e1"].num_token, vocab["rel"].num_token)
    elif args.model == "ggnn_distmult":
        model = GGNNDistMult(args, vocab["e1"].num_token, vocab["rel"].num_token)
    elif args.model == "gcn_distmult":
        model = GCNDistMult(args, vocab["e1"].num_token, vocab["rel"].num_token)
    elif args.model == "gcn_complex":
        model = GCNComplex(args, vocab["e1"].num_token, vocab["rel"].num_token)
    else:
        raise Exception("Unknown model!")

    if args.model in ["ggnn_distmult", "gcn_distmult", "gcn_complex"]:
        graph_path = "examples/pytorch/kg_completion/{}/processed/KG_graph.pt".format(args.data)
        KG_graph = torch.load(graph_path)
        if Config.cuda is True:
            KG_graph = KG_graph.to("cuda")
        else:
            KG_graph = KG_graph.to("cpu")
    else:
        raise RuntimeError('Unknown model {}'.format(args.model))

    if Config.cuda is True:
        model.cuda()

    model_params = torch.load(model_path)
    print(model)
    total_param_size = []
    params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
    for key, size, count in params:
        total_param_size.append(count)
        print(key, size, count)
    print(np.array(total_param_size).sum())
    model.load_state_dict(model_params)
    model.eval()
    ranking_and_hits(model, test_rank_batcher, vocab, "test_evaluation", kg_graph=KG_graph)

    params = [value.numel() for value in model.parameters()]
    print(params)
    print(np.sum(params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link prediction for knowledge graphs")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="input batch size for training (default: 128)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        help="input batch size for testing/validation (default: 128)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train (default: 1000)"
    )
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate (default: 0.003)")
    parser.add_argument(
        "--seed", type=int, default=1234, metavar="S", help="random seed (default: 17)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="kinship",
        help="Dataset to use: {FB15k-237, YAGO3-10, WN18RR, umls, nations, kinship},"
        "default: kinship",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0,
        help="Weight decay value to use in the optimizer. Default: 0.0",
    )
    parser.add_argument(
        "--model", type=str, default="conve", help="Choose from: {conve, distmult, complex}"
    )
    parser.add_argument(
        "--direction_option",
        type=str,
        default="undirected",
        help="Choose from: {undirected, bi_sep, bi_fuse}",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=200, help="The embedding dimension (1D). Default: 200"
    )
    parser.add_argument(
        "--embedding-shape1",
        type=int,
        default=20,
        help="The first dimension of the reshaped 2D embedding. The second"
        "dimension is infered. Default: 20",
    )
    parser.add_argument(
        "--hidden-drop",
        type=float,
        default=0.25,
        help="Dropout for the hidden layer. Default: 0.3.",
    )
    parser.add_argument(
        "--input-drop",
        type=float,
        default=0.2,
        help="Dropout for the input embeddings. Default: 0.2.",
    )
    parser.add_argument(
        "--feat-drop",
        type=float,
        default=0.2,
        help="Dropout for the convolutional features. Default: 0.2.",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.995,
        help="Decay the learning rate by this factor every epoch. Default: 0.995",
    )
    parser.add_argument(
        "--loader-threads",
        type=int,
        default=4,
        help="How many loader threads to use for the batch loaders. Default: 4",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess the dataset. Needs to be executed only once. Default: 4",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a model.")
    parser.add_argument(
        "--use-bias",
        action="store_true",
        help="Use a bias in the convolutional layer. Default: True",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing value to use. Default: 0.1",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=9728,
        help="The side of the hidden layer. The required size changes with the"
        "size of the embeddings. Default: 9728 (embedding size 200).",
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=200,
        help="The side of the hidden layer. The required size changes with the"
        "size of the embeddings. Default: 9728 (embedding size 200).",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="The side of the hidden layer. The required size changes with the"
        "size of the embeddings. Default: 9728 (embedding size 200).",
    )

    args = parser.parse_args()

    # parse console parameters and set global variables
    Config.backend = "pytorch"
    Config.cuda = False
    Config.embedding_dim = args.embedding_dim
    # Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG

    model_name = "{2}_{0}_{1}".format(args.input_drop, args.hidden_drop, args.model)
    model_path = "examples/pytorch/kg_completion/saved_models/{0}_{1}.model".format(
        args.data, model_name
    )

    torch.manual_seed(args.seed)
    main(args, model_path)
