from operator import itemgetter
import pickle as pkl
import torch
from random import randint
import copy



class Tree():
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = []

    def __str__(self, level=0):
        ret = ""
        for child in self.children:
            if isinstance(child, type(self)):
                ret += child.__str__(level+1)
            else:
                ret += "\t"*level + str(child) + "\n"
        return ret

    def add_child(self, c):
        if isinstance(c, type(self)):
            c.parent = self
        self.children.append(c)
        self.num_children = self.num_children + 1

    def to_string(self):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], Tree):
                r_list.append("( " + self.children[i].to_string() + " )")
            else:
                r_list.append(str(self.children[i]))
        return "".join(r_list)

    def to_list(self, tgt_vocab):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], type(self)):
                r_list.append(tgt_vocab.get_symbol_idx("("))
                cl = self.children[i].to_list(tgt_vocab)
                for k in range(len(cl)):
                    r_list.append(cl[k])
                r_list.append(tgt_vocab.get_symbol_idx(")"))
            else:
                r_list.append(self.children[i])
        return r_list

    @staticmethod
    def norm_tree(r_list, tgt_vocab):
        q = [Tree.convert_to_tree(r_list, 0, len(r_list), tgt_vocab)]
        head = 0
        while head < len(q):
            t = q[head]
            exchangeable_symbol_idx_list = tgt_vocab.get_symbol_idx_for_list(['and', 'or', '+', '*'])
            if (t.children[0] in exchangeable_symbol_idx_list):
                k = []
                for i in range(1, len(t.children)):
                    if isinstance(t.children[i], Tree):
                        k.append((t.children[i].to_string(), i))
                    else:
                        k.append((str(t.children[i]), i))
                sorted_t_dict = []
                k.sort(key=itemgetter(0))
                for key1 in k:
                    sorted_t_dict.append(t.children[key1[1]])

                for i in range(t.num_children-1):
                    t.children[i+1] = \
                        sorted_t_dict[i]
            for i in range(len(t.children)):
                if isinstance(t.children[i], Tree):
                    q.append(t.children[i])

            head = head + 1
        return q[0]
    
    @staticmethod
    def convert_to_tree(r_list, i_left, i_right, tgt_vocab):
        t = Tree()
        level = 0
        left = -1
        for i in range(i_left, i_right):
            if r_list[i] == tgt_vocab.get_symbol_idx('('):
                if level == 0:
                    left = i
                level = level + 1
            elif r_list[i] == tgt_vocab.get_symbol_idx(')'):
                level = level - 1
                if level == 0:
                    if i == left+1:
                        c = r_list[i]
                    else:
                        c = Tree.convert_to_tree(r_list, left + 1, i, tgt_vocab)
                    t.add_child(c)
            elif level == 0:
                t.add_child(r_list[i])
        return t


class Vocab():
    def __init__(self):
        self.symbol2idx = {}
        self.idx2symbol = {}
        self.vocab_size = 0

        self.add_symbol('<PAD>')
        self.add_symbol('<START>')
        self.add_symbol('<END>')
        self.add_symbol('<UNK>')
        self.add_symbol('<NON-TERMINAL>')

    def add_symbol(self, s):
        if s not in self.symbol2idx:
            self.symbol2idx[s] = self.vocab_size
            self.idx2symbol[self.vocab_size] = s
            self.vocab_size += 1
        return self.symbol2idx[s]

    def get_symbol_idx(self, s):
        if s not in self.symbol2idx:
            return self.symbol2idx['<U>']
        else:
            print("not reached!")
            return 0
        return self.symbol2idx[s]

    def get_idx_symbol(self, idx):
        if idx not in self.idx2symbol:
            return '<U>'
        return self.idx2symbol[idx]

    def init_from_file(self, fn, min_freq, max_vocab_size):
        print("loading vocabulary file: {}".format(fn))
        with open(fn, "r") as f:
            for line in f:
                l_list = line.strip().split('\t')
                c = int(l_list[1])
                if c >= min_freq:
                    self.add_symbol(l_list[0])
                if self.vocab_size > max_vocab_size:
                    break

    def get_symbol_idx_for_list(self, l):
        r = []
        for i in range(len(l)):
            r.append(self.get_symbol_idx(l[i]))
        return r

class DataLoader():
    def __init__(self, src_vocab_file, tgt_vocab_file, data_file, mode, min_freq, max_vocab_size, batch_size, device):
        self.mode = mode
        self.device = device
        self.data_file = data_file
        self.src_vocab = Vocab()
        self.tgt_vocab = Vocab()

        print("loading vocabulary file...")
        self.src_vocab.init_from_file(src_vocab_file, min_freq, max_vocab_size)
        self.tgt_vocab.init_from_file(tgt_vocab_file, min_freq, max_vocab_size)

        print("loading data file...")
        data = self._get_seq_tree_pair_data()

        print("padding data with batch_size")
        if len(data) % batch_size != 0:
            n = len(data)
            for i in range(batch_size - len(data) % batch_size):
                data.insert(n-i-1, copy.deepcopy(data[n-i-1]))

        self.enc_batch_list = []
        self.enc_len_batch_list = []
        self.dec_batch_list = []
        p = 0
        while p + batch_size <= len(data):
            max_len = len(data[p + batch_size - 1][0])
            m_text = torch.zeros((batch_size, max_len + 2), dtype=torch.long)
            if using_gpu:
                m_text = m_text.cuda()
            enc_len_list = []
            # add <S>
            m_text[:,0] = 0
            for i in range(batch_size):
                w_list = data[p + i][0]
                # reversed order
                for j in range(len(w_list)):
                    #print(max_len+2)
                    m_text[i][j+1] = w_list[len(w_list) - j -1]
                    #m_text[i][j+1] = w_list[j]
                # -- add <E> (for encoder, we need dummy <E> at the end)
                for j in range(len(w_list)+1, max_len+2):
                    m_text[i][j] = 1
                enc_len_list.append(len(w_list)+2)
            self.enc_batch_list.append(m_text)
            self.enc_len_batch_list.append(enc_len_list)

            tree_batch = []
            for i in range(batch_size):
                tree_batch.append(data[p+i][2])
            self.dec_batch_list.append(tree_batch)
            p += batch_size

        self.num_batch = len(self.enc_batch_list)
        assert(len(self.enc_batch_list) == len(self.dec_batch_list))

    def _get_seq_tree_pair_data(self):
        data = []
        with open(self.data_file, "r") as f:
            for line in f:
                l_list = line.split("\t")
                w_list = self.src_vocab.get_symbol_idx_for_list(l_list[0].strip().split(' '))
                r_list = self.tgt_vocab.get_symbol_idx_for_list(l_list[1].strip().split(' '))
                cur_tree = Tree.convert_to_tree(r_list, 0, len(r_list), self.tgt_vocab)
                data.append((w_list, r_list, cur_tree))
        return data

    def random_batch(self):
        p = randint(0,self.num_batch-1)
        return self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]

    def all_batch(self):
        r = []
        for p in range(self.num_batch):
            r.append([self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]])
        return r

def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x
