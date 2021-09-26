from operator import itemgetter
import numpy as np


class Tree:
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = []

    def __str__(self, level=0):
        ret = ""
        for child in self.children:
            if isinstance(child, type(self)):
                ret += child.__str__(level + 1)
            else:
                ret += "\t" * level + str(child) + "\n"
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
                r_list.append(" " + str(self.children[i]) + " ")
        return "".join(r_list)

    def to_text(self, form_manager):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], Tree):
                r_list.append("( " + self.children[i].to_text(form_manager) + " )")
            else:
                r_list.append(" " + form_manager.get_idx_symbol(self.children[i]) + " ")
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
            exchangeable_symbol_idx_list = tgt_vocab.get_symbol_idx_for_list(["and", "or"])
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

                for i in range(t.num_children - 1):
                    t.children[i + 1] = sorted_t_dict[i]
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
                cnt_deleted = 0
                for index in list(range(len(t.children))):
                    if index not in ([item[1] for item in k.items()]):
                        t.children.pop(index - cnt_deleted)
                        t.num_children = t.num_children - 1
                        cnt_deleted += 1

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
            if r_list[i] == tgt_vocab.get_symbol_idx("("):
                if level == 0:
                    left = i
                level = level + 1
            elif r_list[i] == tgt_vocab.get_symbol_idx(")"):
                level = level - 1
                if level == 0:
                    if i == left + 1:
                        c = r_list[i]
                    else:
                        c = Tree.convert_to_tree(r_list, left + 1, i, tgt_vocab)
                    t.add_child(c)
            elif level == 0:
                t.add_child(r_list[i])
        return t


class VocabForAll:
    def __init__(self, in_word_vocab, out_word_vocab, share_vocab):
        self.in_word_vocab = in_word_vocab
        self.out_word_vocab = out_word_vocab
        self.share_vocab = share_vocab

    def get_vocab_size(self):
        if hasattr(self, "share_vocab"):
            return self.share_vocab.vocab_size
        else:
            return self.in_word_vocab.vocab_size + self.out_word_vocab.vocab_size


class Vocab:
    def __init__(
        self,
        lower_case=True,
        pretrained_word_emb_name=None,
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=None,
        embedding_dims=300,
    ):
        self.symbol2idx = {}
        self.idx2symbol = {}
        self.vocab_size = 0

        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        self.non_terminal_token = "<NON-TERMINAL>"

        self.add_symbol(self.pad_token)
        self.add_symbol(self.start_token)
        self.add_symbol(self.end_token)
        self.add_symbol(self.unk_token)
        self.add_symbol(self.non_terminal_token)

        self.lower_case = lower_case
        self.embedding_dims = embedding_dims
        self.embeddings = None

        self.pretrained_word_emb_name = pretrained_word_emb_name
        self.pretrained_word_emb_url = pretrained_word_emb_url
        self.pretrained_word_emb_cache_dir = pretrained_word_emb_cache_dir

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
                l_list = line.strip().split("\t")
                c = int(l_list[1])
                if c >= min_freq:
                    self.add_symbol(l_list[0])
                if self.vocab_size > max_vocab_size:
                    break
        if self.pretrained_word_emb_name == "None" or self.pretrained_word_emb_name is None:
            self.randomize_embeddings(self.embedding_dims)
        else:
            # print("loadding pretrained embedding file in {}".format(self.pretrained_embedding_fn))
            self.load_embeddings(
                pretrained_word_emb_name=self.pretrained_word_emb_name,
                pretrained_word_emb_url=self.pretrained_word_emb_url,
                pretrained_word_emb_cache_dir=self.pretrained_word_emb_cache_dir,
            )

    def init_from_list(self, arr, min_freq=1, max_vocab_size=100000):
        for word_, c_ in arr:
            if c_ >= min_freq:
                self.add_symbol(word_)
            if self.vocab_size > max_vocab_size:
                break
        if self.pretrained_word_emb_name == "None" or self.pretrained_word_emb_name is None:
            self.randomize_embeddings(self.embedding_dims)
        else:
            # print("loadding pretrained embedding file in {}".format(self.pretrained_embedding_fn))
            self.load_embeddings(
                pretrained_word_emb_name=self.pretrained_word_emb_name,
                pretrained_word_emb_url=self.pretrained_word_emb_url,
                pretrained_word_emb_cache_dir=self.pretrained_word_emb_cache_dir,
            )

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

    def load_embeddings(
        self,
        pretrained_word_emb_name="840B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=".vector_cache/",
        dtype=np.float32,
    ):
        """Load pretrained word embeddings for initialization"""
        from .vocab_utils import WordEmbModel

        word_model = WordEmbModel(
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
        )

        word_list = list(self.symbol2idx.keys())

        word_emb, hit, cnt = word_model.get_vecs_by_tokens(
            tokens=word_list, lower_case_backup=self.lower_case
        )

        self.embeddings = word_emb.numpy()
        self.embeddings[self.get_symbol_idx(self.pad_token)] = np.zeros(word_model.dim)

    def randomize_embeddings(self, n_dims, scale=0.08):
        """Use random word embeddings for initialization."""
        vocab_size = self.vocab_size
        shape = (vocab_size, n_dims)
        self.embeddings = np.array(
            np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32
        )
        self.embeddings[self.get_symbol_idx(self.pad_token)] = np.zeros(n_dims)

    def __getitem__(self, item):
        if type(item) is int:
            return self.get_idx_symbol(item)
        return self.get_symbol_idx(item)

    def __len__(self):
        return self.vocab_size


def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x
