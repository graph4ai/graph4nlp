import argparse
import os
import pickle as pkl
import random
import re
import time
import warnings
from .config import get_args, show_args, get_args_for_geo

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim

from .model.graph_encoder import GraphEncoder
from .model.tree_decoder import AttnUnit, DecoderRNN
from .train import do_generate, eval_training
from .utils import data_utils, graph_utils, pretrained_embedding
from .utils.GraphGeneration import GraphGenerator
from .utils.tree import Tree

warnings.filterwarnings('ignore')


def main(opt):

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if opt.graph_generate != 0:
        print("---generating graph input---")
        g_generator = GraphGenerator(opt)
        g_generator.GraphGenerateBegin()
        if opt.graph_generate == 2:
            print("Graph generation done!")
            return
        print("--------------------done------------------")

    print("---loading vocab loader---")
    managers = pkl.load(open("{}\\map.pkl".format(opt.data_dir), "rb"))
    word_manager, form_manager = managers
    print("input vocab size", word_manager.vocab_size)
    print("output vocab size", form_manager.vocab_size)
    print("--------------------done------------------")

    using_gpu = False
    if opt.gpuid > -1:
        using_gpu = True

    # print opt.pretrain_flag
    if opt.pretrain_flag:
        print("--------------generating word pretrained embedding--------------")
        if (os.path.exists("{}/pretrain.pkl".format(opt.vocab_data_dir))):
            print("word embedding has been generated !")
        else:
            print("word embedding generating...")
            pretrained_embedding.generate_embedding_from_glove(opt)
        print("--------------------done------------------")

    encoder = GraphEncoder(opt, word_manager.vocab_size)
    decoder = DecoderRNN(opt, form_manager.vocab_size)
    attention_decoder = AttnUnit(opt, form_manager.vocab_size)

    if using_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        attention_decoder = attention_decoder.cuda()

    for name, param in encoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in attention_decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)

    print("--------------loading train loader----------------")
    train_loader = data_utils.MinibatchLoader(
        opt, 'train', using_gpu, word_manager)
    print("--------------------done------------------")

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    step = 0
    epoch = 0
    optim_state = {"learningRate": opt.learning_rate,
                   "weight_decay":  opt.weight_decay}
    print("using adam")
    encoder_optimizer = optim.Adam(encoder.parameters(
    ),  lr=optim_state["learningRate"], weight_decay=optim_state["weight_decay"])

    decoder_optimizer = optim.Adam(
        decoder.parameters(),  lr=optim_state["learningRate"])

    attention_decoder_optimizer = optim.Adam(
        attention_decoder.parameters(),  lr=optim_state["learningRate"])

    # encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size= (opt.max_epochs//4) * train_loader.num_batch, gamma=0.5)
    # decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=(opt.max_epochs//4) * train_loader.num_batch, gamma=0.5)
    # attention_decoder_scheduler = torch.optim.lr_scheduler.StepLR(attention_decoder_optimizer, step_size=(opt.max_epochs//4) * train_loader.num_batch, gamma=0.5)

    criterion = nn.NLLLoss(size_average=False, ignore_index=0)

    print("Starting training.")
    encoder.train()
    decoder.train()
    attention_decoder.train()
    iterations = opt.max_epochs * train_loader.num_batch
    start_time = time.time()

    best_val_acc = 0

    print("Batch number per Epoch:", train_loader.num_batch)
    opt.print_every = train_loader.num_batch

    loss_to_print = 0
    for i in range(iterations):
        if (i+1) % train_loader.num_batch == 0:
            epoch += 1
        # prev_lr = encoder_scheduler.get_lr()[0]

        # encoder_scheduler.step()
        # decoder_scheduler.step()
        # attention_decoder_scheduler.step()

        # print optim_state["learningRate"]

        # if encoder_scheduler.get_lr()[0] != prev_lr:
        #     print "lr: from {} to {}".format(prev_lr, encoder_scheduler.get_lr()[0])

        epoch = i // train_loader.num_batch
        train_loss = eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer,
                                   decoder_optimizer, attention_decoder_optimizer, criterion, using_gpu, word_manager, form_manager)

        loss_to_print += train_loss
        if i == iterations - 1 or (i+1) % opt.print_every == 0:
            checkpoint = {}
            checkpoint["encoder"] = encoder
            checkpoint["decoder"] = decoder
            checkpoint["attention_decoder"] = attention_decoder
            checkpoint["opt"] = opt
            checkpoint["i"] = i
            checkpoint["epoch"] = epoch
            torch.save(
                checkpoint, "{}/model_g2t".format(opt.checkpoint_dir) + str(i))

        if (i+1) % opt.print_every == 0:
            end_time = time.time()
            print(("{}/{}, train_loss = {}, epochs = {}, time since last print = {}".format(i,
                                                                                           iterations, (loss_to_print / opt.print_every), epoch, (end_time - start_time)/60)))
            loss_to_print = 0
            start_time = time.time()

        if train_loss != train_loss:
            print('loss is NaN.  This usually indicates a bug.')
            break


if __name__ == "__main__":
    start = time.time()
    args = get_args_for_geo()
    show_args(args)

    main(args)
    end = time.time()
    print(("total time: {} minutes\n".format((end - start)/60)))
