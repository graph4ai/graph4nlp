import argparse

from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.utils.config_utils import update_values, get_yaml_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_yaml", type=str,
                        default="examples/pytorch/semantic_parsing/graph2seq/config/new_dependency_graphsage.yaml", help="")
                        # default="examples/pytorch/semantic_parsing/graph2seq/config/new_dynamic_refine.yaml", help="")
                        # default = "examples/pytorch/semantic_parsing/graph2seq/config/new_dynamic.yaml", help = "")

                        # default="examples/pytorch/semantic_parsing/graph2seq/config/new_constituency.yaml", help="")

    parser.add_argument('--word-emb-size', type=int, default=300, help='')
    parser.add_argument("--log-file", type=str, default="examples/pytorch/semantic_parsing/graph2seq/log/ggnn.txt")
    parser.add_argument("--checkpoint-save-path", type=str, default="examples/pytorch/semantic_parsing/graph2seq/save")

    parser.add_argument('--learning-rate', type=float, default=1e-3, help='')
    parser.add_argument("--loss-display-step", type=int, default=100, help=' ')
    parser.add_argument("--eval-display-number", type=int, default=3, help="")
    parser.add_argument("--lr-start-decay-epoch", type=int, default=20, help="")
    parser.add_argument("--lr-decay-rate", type=float, default=0.9)
    parser.add_argument("--lr-decay-per-epoch", type=int, default=5, help="")
    parser.add_argument("--min-lr", type=float, default=1e-3, help="")
    parser.add_argument("--use-gpu", type=float, default=1, help="0 for don't use cuda, 1 for using cuda")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=1236, help="")

    # dataset config
    parser.add_argument("--batch_size", type=int, default=30, help="the size of one mini-batch")
    parser.add_argument("--share-vocab", type=bool, default=True, help="whether to share vocab")
    parser.add_argument("--val_split_ratio", type=float, default=0, help="")
    parser.add_argument("--pretrained_word_emb_file", type=str,
                        default='/raid/ghn/glove/glove.6B.300d.txt', help="")

    parser.add_argument("--beam-size", type=int, default=4, help="the beam size of beam search")

    cfg = parser.parse_args()

    our_args = get_yaml_config(cfg.dataset_yaml)
    template = get_basic_args(graph_construction_name=our_args["graph_construction_name"],
                              graph_embedding_name=our_args["graph_embedding_name"],
                              decoder_name=our_args["decoder_name"])
    update_values(to_args=template, from_args_list=[our_args, vars(cfg)])
    return template

# screen -S ghn_code1 python -m examples.pytorch.summarization.cnn.main_g2s --dataset_yaml examples/pytorch/summarization/cnn/config/new_dependency_graphsage.yaml --word-emb-size 128 --log-file examples/pytorch/summarization/cnn/log/log_gs.log --checkpoint-save-path examples/pytorch/summarization/cnn/save_g2s_3w_copy --learning-rate 0.001 --lr-decay-rate 0.8 --batch_size 30
# screen -S ghn_code2 python -m examples.pytorch.summarization.cnn.main_g2s --dataset_yaml examples/pytorch/summarization/cnn/config/new_dependency_graphsage.yaml --word-emb-size 128 --log-file examples/pytorch/summarization/cnn/log/log_eval.log --checkpoint-save-path examples/pytorch/summarization/cnn/save_g2s_3w_copy --learning-rate 0.001 --lr-decay-rate 0.8 --batch_size 10
# screen -S ghn_code3 python -m examples.pytorch.summarization.cnn.main_g2s --dataset_yaml examples/pytorch/summarization/cnn/config/new_dependency_graphsage.yaml --word-emb-size 128 --log-file examples/pytorch/summarization/cnn/log/log_trans.log --checkpoint-save-path examples/pytorch/summarization/cnn/save_g2s_3w_copy --learning-rate 0.001 --lr-decay-rate 0.8 --batch_size 10
# screen -S ghn_code1 python -m examples.pytorch.summarization.cnn.main_g2s --dataset_yaml examples/pytorch/summarization/cnn/config/new_dependency_gcn.yaml --word-emb-size 128 --log-file examples/pytorch/summarization/cnn/log/log0.log --checkpoint-save-path examples/pytorch/summarization/cnn/save_g2s_3w_copy --learning-rate 0.001 --lr-decay-rate 0.8 --batch_size 10
# --dataset_yaml config/new_dependency_gcn.yaml --word-emb-size 128 --log-file log/log0.log --checkpoint-save-path cnn/save_g2s_3w_copy --learning-rate 0.001 --lr-decay-rate 0.8