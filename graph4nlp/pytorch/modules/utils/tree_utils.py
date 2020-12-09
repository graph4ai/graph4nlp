from operator import itemgetter
import pickle as pkl
import torch
from random import randint
import numpy as np
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
                r_list.append(" "+str(self.children[i])+" ")
        return "".join(r_list)

    def to_text(self, form_manager):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], Tree):
                r_list.append("( " + self.children[i].to_text(form_manager) + " )")
            else:
                r_list.append(" "+form_manager.get_idx_symbol(self.children[i])+" ")
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
            exchangeable_symbol_idx_list = tgt_vocab.get_symbol_idx_for_list(
                ['and', 'or'])
            if len(t.children) > 0 and (t.children[0] in exchangeable_symbol_idx_list):
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
        return q[0].to_list(tgt_vocab)

    @staticmethod
    def deduplicate_tree(r_list, tgt_vocab):
        q = [Tree.convert_to_tree(r_list, 0, len(r_list), tgt_vocab)]
        head = 0
        while head < len(q):
            t = q[head]
            if len(t.children) > 0:
                k = {}
                for i in range(len(t.children)):
                    if isinstance(t.children[i], Tree):
                        k[t.children[i].to_string().strip()] = i
                    else:
                        k[str(t.children[i]).strip()] = i
                for index in (list(range(len(t.children)))):
                    if index not in ([item[1] for item in k.items()]):
                        t.children.pop(index)
                        t.num_children = t.num_children - 1

            for i in range(len(t.children)):
                if isinstance(t.children[i], Tree):
                    q.append(t.children[i])
            head = head + 1
        return q[0].to_list(tgt_vocab)

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
                        c = Tree.convert_to_tree(
                            r_list, left + 1, i, tgt_vocab)
                    t.add_child(c)
            elif level == 0:
                t.add_child(r_list[i])
        return t

class VocabForAll():
    def __init__(self, in_word_vocab, out_word_vocab, share_vocab):
        self.in_word_vocab = in_word_vocab
        self.out_word_vocab = out_word_vocab
        self.share_vocab = share_vocab

class Vocab():
    def __init__(self, lower_case=True, pretrained_embedding_fn=None, embedding_dims=300):
        self.symbol2idx = {}
        self.idx2symbol = {}
        self.vocab_size = 0

        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'
        self.non_terminal_token = '<NON-TERMINAL>'

        self.add_symbol(self.pad_token)
        self.add_symbol(self.start_token)
        self.add_symbol(self.end_token)
        self.add_symbol(self.unk_token)
        self.add_symbol(self.non_terminal_token)

        self.lower_case = lower_case
        self.embedding_dims = embedding_dims
        self.embeddings = None

        self.pretrained_embedding_fn = pretrained_embedding_fn

    def add_symbol(self, s):
        if s not in self.symbol2idx:
            self.symbol2idx[s] = self.vocab_size
            self.idx2symbol[self.vocab_size] = s
            self.vocab_size += 1
        return self.symbol2idx[s]

    def get_symbol_idx(self, s):
        if s not in self.symbol2idx:
            return self.get_symbol_idx(self.unk_token)
        else:
            return self.symbol2idx[s]

    def get_idx_symbol(self, idx):
        if idx not in self.idx2symbol:
            return self.unk_token
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
        if self.pretrained_embedding_fn is None:
            self.randomize_embeddings(self.embedding_dims)
        else:
            print("loadding pretrained embedding file in {}".format(self.pretrained_embedding_fn))
            self.load_embeddings(self.pretrained_embedding_fn)

    def init_from_list(self, arr, min_freq=1, max_vocab_size=100000):
        for word_, c_ in arr:
            if c_ >= min_freq:
                self.add_symbol(word_)
            if self.vocab_size > max_vocab_size:
                break
        if self.pretrained_embedding_fn is None:
            self.randomize_embeddings(self.embedding_dims)
        else:
            print("loadding pretrained embedding file in {}".format(self.pretrained_embedding_fn))
            self.load_embeddings(self.pretrained_embedding_fn)

    def get_symbol_idx_for_list(self, l):
        r = []
        for i in range(len(l)):
            r.append(self.get_symbol_idx(l[i]))
        return r
    
    def get_idx_symbol_for_list(self, l):
        l = np.array(l)
        r = []
        for i in range(len(l)):
            r.append(self.get_idx_symbol(l[i]))
        return r

    def load_embeddings(self, file_path, scale=0.08, dtype=np.float32):
        """Load pretrained word embeddings for initialization"""
        hit_words = set()
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for line in f:
                line = line.split()
                word = line[0].decode('utf-8')
                if self.lower_case:
                    word = word.lower()

                idx = self.get_symbol_idx(word)
                if idx == self.get_symbol_idx(self.unk_token) or idx in hit_words:
                    continue

                vec = np.array(line[1:], dtype=dtype)
                if self.embeddings is None:
                    n_dims = len(vec)
                    self.embeddings = np.array(np.random.uniform(low=-scale, high=scale, size=(vocab_size, n_dims)),
                                               dtype=dtype)
                    self.embeddings[self.get_symbol_idx(self.pad_token)] = np.zeros(n_dims)
                self.embeddings[idx] = vec
                hit_words.add(idx)
        print('Pretrained word embeddings hit ratio: {}'.format(len(hit_words) / len(self.index2word)))

    def randomize_embeddings(self, n_dims, scale=0.08):
        """Use random word embeddings for initialization."""
        vocab_size = self.vocab_size
        shape = (vocab_size, n_dims)
        self.embeddings = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)
        self.embeddings[self.get_symbol_idx(self.pad_token)] = np.zeros(n_dims)

    def __getitem__(self, item):
        if type(item) is int:
            return self.get_idx_symbol(item)
        return self.get_symbol_idx(item)

    def __len__(self):
        return self.vocab_size


class DataLoaderForSeqEncoder():
    def __init__(self, data_dir, use_copy, src_vocab_file, tgt_vocab_file, data_file, mode, min_freq, max_vocab_size, batch_size, device):
        self.mode = mode
        self.device = device
        self.data_file = data_file
        self.src_vocab = Vocab()
        self.tgt_vocab = Vocab()
        self.use_copy = use_copy

        if use_copy:
            share_vocab_file_path = self.generate_share_vocab_file(data_dir, src_vocab_file, tgt_vocab_file, min_freq, max_vocab_size)

        print("loading vocabulary file...")
        if not use_copy:
            self.src_vocab.init_from_file(src_vocab_file, min_freq, max_vocab_size)
            self.tgt_vocab.init_from_file(tgt_vocab_file, 0, max_vocab_size)
        else:
            self.src_vocab.init_from_file(share_vocab_file_path, min_freq, max_vocab_size)
            self.tgt_vocab = self.src_vocab
            self.share_vocab = self.src_vocab
            
        print("loading data file...")
        self.data = self._get_seq_tree_pair_data()

        if self.mode == "test":
            assert(batch_size == 1)

        print("padding data with batch_size")
        if len(self.data) % batch_size != 0:
            n = len(self.data)
            for i in range(batch_size - len(self.data) % batch_size):
                self.data.insert(n-i-1, copy.deepcopy(self.data[n-i-1]))

        self.enc_batch_list = []
        self.enc_len_batch_list = []
        self.dec_batch_list = []

        p = 0
        while p + batch_size <= len(self.data):
            max_len = len(self.data[p + batch_size - 1][0])
            m_text = torch.zeros((batch_size, max_len + 2), dtype=torch.long)
            m_text = to_cuda(m_text, device)

            enc_len_list = []
            # add start token
            m_text[:, 0] = self.src_vocab.get_symbol_idx(self.src_vocab.start_token)
            for i in range(batch_size):
                w_list = self.data[p + i][0]
                # print(w_list)
                # reversed order
                for j in range(len(w_list)):
                    m_text[i][j+1] = w_list[len(w_list) - j - 1]
                # add end token
                add_end = 0
                for j in range(len(w_list)+1, max_len+2):
                    if add_end == 0:
                        m_text[i][j] = self.src_vocab.get_symbol_idx(self.src_vocab.end_token)
                        add_end = 1
                    else:
                        m_text[i][j] = self.src_vocab.get_symbol_idx(self.src_vocab.pad_token)
                # print(m_text[i], '\n')
                enc_len_list.append(len(w_list)+2)
            self.enc_batch_list.append(m_text)
            self.enc_len_batch_list.append(enc_len_list)

            tree_batch = []
            for i in range(batch_size):
                tree_batch.append(self.data[p+i][2])
            self.dec_batch_list.append(tree_batch)
            p += batch_size

        self.num_batch = len(self.enc_batch_list)
        assert(len(self.enc_batch_list) == len(self.dec_batch_list))
        print("---Done---")

    def generate_share_vocab_file(self, data_dir, src_vocab_file, tgt_vocab_file, min_freq, max_vocab_size):
        self.src_vocab.init_from_file(src_vocab_file, min_freq, max_vocab_size)
        self.tgt_vocab.init_from_file(tgt_vocab_file, 0, max_vocab_size)

        share_vocab = {}
        num = np.max([self.src_vocab.vocab_size, self.tgt_vocab.vocab_size]) + min_freq + 1

        for word, freq in self.src_vocab.symbol2idx.items():
            freq = num-freq
            if word not in share_vocab.keys():
                share_vocab[word] = freq
            else:
                share_vocab[word] += freq

        for word, freq in self.tgt_vocab.symbol2idx.items():
            freq = num-freq
            if word not in share_vocab.keys():
                share_vocab[word] = freq
            else:
                share_vocab[word] += freq

        share_vocab = sorted(share_vocab.items(), key=lambda d:d[1], reverse = True)[5:]
        with open(data_dir + 'share_vocab.txt','w') as f:
            for k, v in share_vocab:
                f.write(k+'\t'+str(v)+'\n')
        return (data_dir + 'share_vocab.txt')


    def _get_seq_tree_pair_data(self):
        data = []
        with open(self.data_file, "r") as f:
            for line in f:
                l_list = line.strip().split("\t")
                w_list = self.src_vocab.get_symbol_idx_for_list(
                    l_list[0].strip().split(' '))
                r_list = self.tgt_vocab.get_symbol_idx_for_list(
                    l_list[1].strip().split(' '))
                cur_tree = Tree.convert_to_tree(
                    r_list, 0, len(r_list), self.tgt_vocab)
                data.append((w_list, r_list, cur_tree))
        return data

    def random_batch(self):
        p = randint(0, self.num_batch-1)
        return self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]

    def all_batch(self):
        r = []
        for p in range(self.num_batch):
            r.append([self.enc_batch_list[p],
                      self.enc_len_batch_list[p], self.dec_batch_list[p]])
        return r


class DataLoaderForGraphEncoder():
    def __init__(self, use_copy, use_share_vocab, data, dataset, mode, batch_size, device):
        self.mode = mode
        self.device = device
        self.use_copy = use_copy
        self.batch_size = batch_size
        self.src_vocab = dataset.src_vocab_model
        self.tgt_vocab = dataset.tgt_vocab_model
        if use_share_vocab and self.src_vocab.vocab_size == self.tgt_vocab.vocab_size:
            self.share_vocab = self.src_vocab
        
        self.data = [(item.graph, item.output_text, item.output_tree) for item in data]

        if self.mode == "test":
            assert(batch_size == 1)

        if self.mode == "train" and len(self.data) % batch_size != 0:
            n = len(self.data)
            for i in range(batch_size - len(self.data) % batch_size):
                self.data.insert(n-i-1, copy.deepcopy(self.data[n-i-1]))

        self.enc_batch_list = []
        self.enc_len_batch_list = []
        self.original_target_tree_list = []
        self.dec_batch_list = []

        p = 0
        while p + batch_size <= len(self.data):
            vectorized_batch_graph = []
            tree_batch = []
            original_target_tree = []
            enc_len_batch = 0

            for i in range(batch_size):
                graph_item = self.data[p + i][0]
                vectorized_batch_graph.append(graph_item)
                enc_len_batch += graph_item.get_node_num()
                original_target_tree.append(self.data[p+i][1])
                tree_batch.append(self.data[p+i][2])
            
            self.enc_batch_list.append(vectorized_batch_graph)
            self.enc_len_batch_list.append(enc_len_batch)
            self.original_target_tree_list.append(original_target_tree)
            self.dec_batch_list.append(tree_batch)
            p += batch_size

        self.num_batch = len(self.enc_batch_list)
        assert(len(self.enc_batch_list) == len(self.dec_batch_list))


    def random_batch(self):
        p = randint(0, self.num_batch-1)
        return self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p], self.original_target_tree_list[p]
    
    def all_batch(self):
        r = []
        for p in range(self.num_batch):
            r.append([self.enc_batch_list[p],
                      self.enc_len_batch_list[p], self.dec_batch_list[p]], self.original_target_tree_list[p])
        return r


def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x
