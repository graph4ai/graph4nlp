import logging
import math
import os
import pickle
import torch
import torchtext.vocab as vocab
from torch.optim.lr_scheduler import LambdaLR

from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps
    following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows
    cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def save_config(opt, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "opt.pkl"), "wb") as f:
        pickle.dump(opt, f)


def get_log(log_file):
    log_folder = os.path.split(log_file)[0]
    os.makedirs(log_folder, exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_glove_weights(vocabulary: Vocab, dim=300):
    glove = vocab.GloVe(name="6B", dim=dim)
    vocab_size = vocabulary.get_vocab_size()
    weight = torch.randn(vocab_size, dim)
    for i, word in enumerate(vocabulary.index2word):
        glove_rep = glove.get_vecs_by_tokens(word)
        weight[i, :] = glove_rep
    return weight


def wordid2str(word_ids, vocab: Vocab):
    ret = []
    assert len(word_ids.shape) == 2, print(word_ids.shape)
    for i in range(word_ids.shape[0]):
        id_list = word_ids[i, :]
        ret_inst = []
        for j in range(id_list.shape[0]):
            if id_list[j] == vocab.EOS:
                break
            token = vocab.getWord(id_list[j])
            ret_inst.append(token)
        ret.append(" ".join(ret_inst))
    return ret
