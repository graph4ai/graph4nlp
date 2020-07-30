from argparse import ArgumentParser


def show_args(opt):
    print("---------------Configuration------------------")
    print("Train or just graph generation:", opt.graph_generate)
    print("Data Directory:", opt.data_dir)
    print("Random Seed:", opt.seed)
    print("Directory checkpoints are saved:", opt.checkpoint_dir)
    print("How many epochs between printing the loss:", opt.print_every)

    print("---------------Network Parameter--------------")
    print("RNN hidden layer size:", opt.rnn_size)
    print("Word embedding size", opt.emb_dim)
    print("Graph encoder hop size:", opt.sample_layer_size)

    print("---------------Training Setting---------------")
    print("Batch size:", opt.batch_size)
    print("Max epochs:", opt.max_epochs)
    print("Learning rate:", opt.learning_rate)
    print("Weight decay:", opt.weight_decay)
    print()
    print("Dropout for GraphEncoder in input:", opt.dropout_en_in)
    print("Dropout for GraphEncoder in output(cell state):", opt.dropout_en_out)

    print("Dropout for Decoder in input:", opt.dropout_de_in)
    print("Dropout for Decoder in output(cell state):", opt.dropout_de_out)

    print("Dropout in attention layer:", opt.dropout_for_predict)
    print()

    print("---------------Some Options-------------------")
    print("Whether use pretrained embedding:", (opt.pretrain_flag))
    print("Whether Regenerate Graph input:", (opt.graph_generate))

    print("----------------------------------------------")


def get_args():
    parser = ArgumentParser(description="parser")

    parser.add_argument('-gpuid', type=int, default=0,
                        help='which gpu to use. -1 = use CPU')

    parser.add_argument('-data_dir', type=str,
                        default= r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\GraphConstruction", help='graph and tree data_dir')

    parser.add_argument('-seed', type=int, default=123,
                        help='torch manual random number generator seed')
    parser.add_argument('-checkpoint_dir', type=str, default='../experiments',
                        help='output directory where checkpoints and training states are saved')
    parser.add_argument('-print_every', type=int, default=-1,
                        help='how many steps/minibatches between printing out the loss, -1 for per epoch')
    parser.add_argument('-rnn_size', type=int, default=300,
                        help='size of LSTM internal state')
    parser.add_argument('-emb_dim', type=int, default=300,
                        help='size of word embedding size')

    parser.add_argument('-num_layers', type=int, default=1,
                        help='number of layers in the LSTM')

    parser.add_argument('-dropout_en_in', type=float,
                        default=0.1, help='dropout for encoder, input')
    parser.add_argument('-dropout_en_out', type=float,
                        default=0.3, help='dropout for encoder, output')

    parser.add_argument('-dropout_de_in', type=float,
                        default=0.1, help='dropout for decoder, input')
    parser.add_argument('-dropout_de_out', type=float,
                        default=0.3, help='dropout for decoder, output')
    parser.add_argument('-dropout_for_predict', type=float, default=0.1,
                        help='dropout used in attention decoder, in prediction')

    parser.add_argument('-dropoutagg', type=float, default=0,
                        help='dropout for regularization, used after each aggregator. 0 = no dropout')

    parser.add_argument('-dec_seq_length', type=int, default=220,
                        help='number of timesteps to unroll for')

    parser.add_argument('-batch_size', type=int, default=20,
                        help='number of sequences to train on in parallel')

    parser.add_argument('-max_epochs', type=int, default=800,
                        help='number of full passes through the training data')

    parser.add_argument('-learning_rate', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('-init_weight', type=float,
                        default=0.08, help='initailization weight')

    parser.add_argument('-weight_decay', type=float,
                        default=1e-5, help='decay rate for adam')

    parser.add_argument('-grad_clip', type=int, default=5,
                        help='clip gradients at this value')

    ''' some arguments of graph encoder '''
    parser.add_argument('-graph_encode_direction', type=str,
                        default='uni', help='graph encode direction: bi or uni')
    parser.add_argument('-sample_size_per_layer', type=int,
                        default=10, help='sample_size_per_layer')
    parser.add_argument('-sample_layer_size', type=int,
                        default=2, help='sample_layer_size')
    parser.add_argument('-concat', type=int, default=1,
                        help='concat in aggregators settings')
    parser.add_argument('-separate_att', type=int,
                        default=1, help='separate attention mechanism')

    ''' Pretrain embedding '''
    parser.add_argument('-pretrain_flag', type=int, default=1)
    parser.add_argument('-vocab_data_dir', type=str,
                        default='../data/TextData', help='data path')
    parser.add_argument('-pretrained_embedding_text', type=str,
                        default="/home/lishucheng/projects/Tools-and-Resources/glove/glove.6B.300d.txt")
    # parser.add_argument('-pretrained_embedding_text', type=str,
    #                     default="/Users/scheng_lee/Documents/glove/glove.6B.300d.txt")

    ''' Graph generation '''
    parser.add_argument('-graph_generate', type=int, default=0,
                        help='0 for just train, 1 for graph generation and train, 2 for just graph generation')
    parser.add_argument('-source_data_dir', type=str,
                        default='../data/TextData/', help='data path')

    parser.add_argument('-output_data_dir', type=str,
                        default='../data/GraphConstruction/', help='data path')

    parser.add_argument('-min_freq', type=int,
                        default=2)
    parser.add_argument('-max_vocab_size', type=int,
                        default=15000)
    parser.add_argument('-parse_chinese_or_english', type=int,
                        default=0, help='0 for english, 1 for chinese')
    parser.add_argument('-generate_mutiple_graph', type=int,
                        default=0, help='0 for single, 1 for multiple')
    args = parser.parse_args()
    return args


def get_args_for_geo():
    parser = ArgumentParser(description="parser")

    parser.add_argument('-gpuid', type=int, default=0,
                        help='which gpu to use. -1 = use CPU')

    parser.add_argument('-data_dir', type=str,
                        default= r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\GraphConstruction", help='graph and tree data_dir')

    parser.add_argument('-seed', type=int, default=123,
                        help='torch manual random number generator seed')
    parser.add_argument('-checkpoint_dir', type=str, default='../experiments',
                        help='output directory where checkpoints and training states are saved')
    parser.add_argument('-print_every', type=int, default=-1,
                        help='how many steps/minibatches between printing out the loss, -1 for per epoch')
    parser.add_argument('-rnn_size', type=int, default=300,
                        help='size of LSTM internal state')
    parser.add_argument('-emb_dim', type=int, default=300,
                        help='size of word embedding size')

    parser.add_argument('-num_layers', type=int, default=1,
                        help='number of layers in the LSTM')
    parser.add_argument('-dropout_en_in', type=float,
                        default=0, help='dropout for encoder, input')
    parser.add_argument('-dropout_en_out', type=float,
                        default=0, help='dropout for encoder, output')

    parser.add_argument('-dropout_de_in', type=float,
                        default=0.1, help='dropout for decoder, input')
    parser.add_argument('-dropout_de_out', type=float,
                        default=0.3, help='dropout for decoder, output')
    parser.add_argument('-dropout_for_predict', type=float, default=0.1,
                        help='dropout used in attention decoder, in prediction')

    parser.add_argument('-dropoutagg', type=float, default=0,
                        help='dropout for regularization, used after each aggregator. 0 = no dropout')

    parser.add_argument('-para_for_sibling', type=float, default=0.001)

    parser.add_argument('-dec_seq_length', type=int, default=100,
                        help='number of timesteps to unroll for')

    parser.add_argument('-batch_size', type=int, default=20,
                        help='number of sequences to train on in parallel')

    parser.add_argument('-max_epochs', type=int, default=1000,
                        help='number of full passes through the training data')

    parser.add_argument('-learning_rate', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('-init_weight', type=float,
                        default=0.08, help='initailization weight')

    parser.add_argument('-weight_decay', type=float,
                        default=1e-5, help='decay rate for adam')

    parser.add_argument('-grad_clip', type=int, default=5,
                        help='clip gradients at this value')

    ''' some arguments of graph encoder '''
    parser.add_argument('-graph_encode_direction', type=str,
                        default='uni', help='graph encode direction: bi or uni')
    parser.add_argument('-sample_size_per_layer', type=int,
                        default=10, help='sample_size_per_layer')
    parser.add_argument('-sample_layer_size', type=int,
                        default=2, help='sample_layer_size')
    parser.add_argument('-concat', type=int, default=1,
                        help='concat in aggregators settings')
    parser.add_argument('-separate_att', type=int,
                        default=1, help='separate attention mechanism')

    ''' Pretrain embedding '''
    parser.add_argument('-pretrain_flag', type=int, default=0)
    parser.add_argument('-vocab_data_dir', type=str,
                        default= r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\TextData", help='data path')
    parser.add_argument('-pretrained_embedding_text', type=str,
                        default="/home/lishucheng/projects/Tools-and-Resources/glove/glove.6B.300d.txt")
    # parser.add_argument('-pretrained_embedding_text', type=str,
    #                     default="/Users/scheng_lee/Documents/glove/glove.6B.300d.txt")

    ''' Graph generation '''
    parser.add_argument('-graph_generate', type=int, default=0,
                        help='0 for just train, 1 for graph generation and train, 2 for just graph generation')
    parser.add_argument('-source_data_dir', type=str,
                        default= r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\TextData", help='data path')

    parser.add_argument('-output_data_dir', type=str,
                        default=r"C:\Users\shuchengli\Desktop\Code\g4nlp\graph4nlp\graph4nlp\pytorch\test\generation\tree_decoder\data\GraphConstruction", help='data path')

    parser.add_argument('-min_freq', type=int,
                        default=2)
    parser.add_argument('-max_vocab_size', type=int,
                        default=15000)
    parser.add_argument('-parse_chinese_or_english', type=int,
                        default=0, help='0 for english, 1 for chinese')
    parser.add_argument('-generate_mutiple_graph', type=int,
                        default=0, help='0 for single, 1 for multiple')
    args = parser.parse_args()
    return args
