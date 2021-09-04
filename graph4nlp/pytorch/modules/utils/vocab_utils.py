import os
import pickle
import re
import warnings
from collections import Counter
from functools import lru_cache
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from torchtext.vocab import GloVe, Vectors

from . import constants

word_detector = re.compile("\w")  # noqa


class VocabModel(object):
    """Vocab model builder.

    Parameters
    ----------
    data_set: iterable
        A list of instances where each instance is a list of str.
    tokenizer: function, optional
        Word tokenization function, default: ``nltk.tokenize.word_tokenize``.
    max_word_vocab_size: int, optional
        Maximal word vocab size, default: ``None``.
    min_word_vocab_freq: int, optional
        Minimal word vocab frequency, default: ``1``.
    pretrained_word_emb_name: str, optional, default="840B"
        The name of pretrained word embedding in ``torchtext``.
        If it is set ``None``, we will randomly set the initial word embedding values.
    pretrained_word_emb_url: str optional, default: ``None``
        The url for downloading pretrained word embedding.
        Note that we only prepare the default ``url`` for English with ``pretrained_word_emb_name``
        as ``"42B"``, ``"840B"``, 'twitter.27B' and '6B'.
    target_pretrained_word_emb_name: str, optional, default=None
        The name of pretrained word embedding in ``torchtext`` for target language.
        If it is set ``None``, we will use ``pretrained_word_emb_name``.
    target_pretrained_word_emb_url: str optional, default: ``None``
        The url for downloading pretrained word embedding for target language.
        Note that we only prepare the default ``url`` for English with ``pretrained_word_emb_name``
        as ``"42B"``, ``"840B"``, 'twitter.27B' and '6B'.
    pretrained_word_emb_cache_dir: str, optional, default: ``".vector_cache/"``
        The path of directory saving the temporary word embedding file.
    word_emb_size: int, optional
        Word embedding size, default: ``None``.
    share_vocab : boolean
        Specify whether to share vocab between input and output text, default: ``True``.

    Examples
    -------
    # Build a vocab model from scratch
    >>> vocab_model = VocabModel([['I like nlp.', 'Same here!'],
                        ['I like graph.', 'Same here!']],
                        max_word_vocab_size=None,
                        min_word_vocab_freq=1,
                        word_emb_size=300)
    >>> print(vocab_model.word_vocab.get_vocab_size())

    # Restore a vocab model from disk if exists or build one from scratch.
    >>> vocab_model = VocabModel.build('vocab_model.pkl', [['I like nlp.', 'Same here!'],
                            ['I like graph.', 'Same here!']],
                            max_word_vocab_size=None,
                            min_word_vocab_freq=1,
                            word_emb_size=300)
    >>> print(vocab_model.word_vocab.get_vocab_size())
    """

    def __init__(
        self,
        data_set=None,
        tokenizer=word_tokenize,
        lower_case=True,
        max_word_vocab_size=None,
        min_word_vocab_freq=1,
        pretrained_word_emb_name="840B",
        pretrained_word_emb_url=None,
        target_pretrained_word_emb_name=None,
        target_pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=".vector_cache/",
        #  pretrained_word_emb_file=None,
        word_emb_size=None,
        share_vocab=True,
    ):
        super(VocabModel, self).__init__()
        self.tokenizer = tokenizer

        print("Building vocabs...")
        all_words = VocabModel.collect_vocabs(
            data_set, self.tokenizer, lower_case=lower_case, share_vocab=share_vocab
        )
        # print('Number of words: {}'.format(len(all_words)))
        if share_vocab:
            in_all_words, out_all_words = all_words, None
            if pretrained_word_emb_name is not None and target_pretrained_word_emb_name is not None:
                warnings.warn(
                    "Warning: share vocabulary for source and target language but use different"
                    "pretrained word embeddings"
                )
        else:
            in_all_words, out_all_words = all_words
            if pretrained_word_emb_name is not None and target_pretrained_word_emb_name is None:
                warnings.warn(
                    "Warning: use separate vocabularies for source and target language but same"
                    "pretrained word embeddings"
                )

        self.in_word_vocab = Vocab(lower_case=lower_case, tokenizer=self.tokenizer)
        self.in_word_vocab.build_vocab(
            in_all_words, max_vocab_size=max_word_vocab_size, min_vocab_freq=min_word_vocab_freq
        )

        if pretrained_word_emb_name is not None:
            self.in_word_vocab.load_embeddings(
                pretrained_word_emb_name=pretrained_word_emb_name,
                pretrained_word_emb_url=pretrained_word_emb_url,
                pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
                pretrained_word_emb_dim=word_emb_size,
            )
            print("Using pretrained word embeddings")
        else:
            self.in_word_vocab.randomize_embeddings(word_emb_size)

        if out_all_words is not None:
            self.out_word_vocab = Vocab(lower_case=lower_case, tokenizer=self.tokenizer)
            self.out_word_vocab.build_vocab(
                out_all_words,
                max_vocab_size=max_word_vocab_size,
                min_vocab_freq=min_word_vocab_freq,
            )

            if target_pretrained_word_emb_name is not None:
                self.out_word_vocab.load_embeddings(
                    pretrained_word_emb_name=target_pretrained_word_emb_name,
                    pretrained_word_emb_url=target_pretrained_word_emb_url,
                    pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
                    pretrained_word_emb_dim=word_emb_size,
                )
                print("Using pretrained word embeddings")
            elif pretrained_word_emb_name is not None:
                self.out_word_vocab.load_embeddings(
                    pretrained_word_emb_name=pretrained_word_emb_name,
                    pretrained_word_emb_url=pretrained_word_emb_url,
                    pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
                    pretrained_word_emb_dim=word_emb_size,
                )
                print("Using pretrained word embeddings")
            else:
                self.out_word_vocab.randomize_embeddings(word_emb_size)
        else:
            self.out_word_vocab = self.in_word_vocab

        if share_vocab:
            print("[ Initialized word embeddings: {} ]".format(self.in_word_vocab.embeddings.shape))
        else:
            print("[ Using separate word vocabs for input & output text ]")
            print(
                "[ Initialized input word embeddings: {} ]".format(
                    self.in_word_vocab.embeddings.shape
                )
            )
            print(
                "[ Initialized output word embeddings: {} ]".format(
                    self.out_word_vocab.embeddings.shape
                )
            )

    @classmethod
    def build(
        cls,
        saved_vocab_file,
        data_set=None,
        tokenizer=word_tokenize,
        lower_case=True,
        max_word_vocab_size=None,
        min_word_vocab_freq=1,
        pretrained_word_emb_name="840B",
        pretrained_word_emb_url=None,
        target_pretrained_word_emb_name=None,
        target_pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=".vector_cache/",
        word_emb_size=None,
        share_vocab=True,
    ):
        """Static method for loading a VocabModel from disk.

        Parameters:
        -------
        saved_vocab_file : str
            Path to the saved vocab file.
        data_set: iterable
            A list of instances where each instance is a list of str.
        tokenizer: function, optional
            Word tokenization function, default: nltk.tokenize.word_tokenize.
        max_word_vocab_size: int, optional
            Maximal word vocab size, default: ``None``.
        min_word_vocab_freq: int, optional
            Minimal word vocab frequency, default: ``1``.
        pretrained_word_emb_name: str, optional, default="840B"
            The name of pretrained word embedding in ``torchtext``.
            If it is set ``None``, we will randomly set the initial word embedding values.
        pretrained_word_emb_url: str optional, default: ``None``
            The url for downloading pretrained word embedding.
            Note that we only prepare the default ``url`` for English with
            ``pretrained_word_emb_name`` as ``"42B"``, ``"840B"``, 'twitter.27B' and '6B'.
        target_pretrained_word_emb_name: str, optional, default=None
            The name of pretrained word embedding in ``torchtext`` for target language.
            If it is set ``None``, we will use ``pretrained_word_emb_name``.
        target_pretrained_word_emb_url: str optional, default: ``None``
            The url for downloading pretrained word embedding for target language.
            Note that we only prepare the default ``url`` for English with
            ``pretrained_word_emb_name`` as ``"42B"``, ``"840B"``, 'twitter.27B' and '6B'.
        pretrained_word_emb_cache_dir: str, optional, default: ``".vector_cache/"``
            The path of directory saving the temporary word embedding file.
        word_emb_size: int, optional
            Word embedding size, default: ``None``.
        share_vocab : boolean
            Specify whether to share vocab between input and output text, default: ``True``.

        Returns:
        -------
        VocabModel
            Loaded Vocabulary.
        """
        if os.path.exists(saved_vocab_file):
            print("Loading pre-built vocab model stored in {}".format(saved_vocab_file))
            with open(saved_vocab_file, "rb") as f:
                vocab_model = pickle.load(f)

        else:
            vocab_model = cls(
                data_set=data_set,
                tokenizer=tokenizer,
                max_word_vocab_size=max_word_vocab_size,
                min_word_vocab_freq=min_word_vocab_freq,
                pretrained_word_emb_name=pretrained_word_emb_name,
                pretrained_word_emb_url=pretrained_word_emb_url,
                pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
                target_pretrained_word_emb_name=target_pretrained_word_emb_name,
                target_pretrained_word_emb_url=target_pretrained_word_emb_url,
                word_emb_size=word_emb_size,
                lower_case=lower_case,
                share_vocab=share_vocab,
            )
            print("Saving vocab model to {}".format(saved_vocab_file))
            pickle.dump(vocab_model, open(saved_vocab_file, "wb"))

        return vocab_model

    @staticmethod
    def collect_vocabs(all_instances, tokenizer, lower_case=True, share_vocab=True):
        """Count vocabulary tokens."""
        if share_vocab:
            all_words = Counter()
        else:
            all_words = [Counter(), Counter()]

        for instance in all_instances:
            extracted_tokens = instance.extract()
            if share_vocab:
                all_words.update(extracted_tokens)
            else:
                all_words[0].update(extracted_tokens[0])
                all_words[1].update(extracted_tokens[1])

        return all_words


class WordEmbModel(Vectors):
    def __init__(
        self,
        pretrained_word_emb_name="840B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=".vector_cache/",
        pretrained_word_emb_dim=300,
    ):

        if pretrained_word_emb_name in GloVe.url.keys() and pretrained_word_emb_url is None:
            url = GloVe.url[pretrained_word_emb_name]
            name = "glove.{}.{}d.txt".format(pretrained_word_emb_name, str(pretrained_word_emb_dim))
        else:
            url = pretrained_word_emb_url
            name = pretrained_word_emb_name
        super().__init__(name=name, cache=pretrained_word_emb_cache_dir, url=url)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]], True
        else:
            return (
                torch.Tensor(
                    np.array(
                        np.random.uniform(low=-0.08, high=0.08, size=(self.dim)), dtype=np.float
                    )
                ),
                False,
            )

    def get_vecs_by_tokens(self, tokens, lower_case_backup=False):
        """Look up embedding vectors of tokens.

        Arguments:
            tokens: a token or a list of tokens. if `tokens` is a string,
                returns a 1-D tensor of shape `self.dim`; if `tokens` is a
                list of strings, returns a 2-D tensor of shape=(len(tokens),
                self.dim).
            lower_case_backup : Whether to look up the token in the lower case.
                If False, each token in the original case will be looked up;
                if True, each token in the original case will be looked up first,
                if not found in the keys of the property `stoi`, the token in the
                lower case will be looked up. Default: False.

        Examples:
            >>> examples = ['chip', 'baby', 'Beautiful']
            >>> vec = text.vocab.GloVe(name='6B', dim=50)
            >>> ret = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
        """
        to_reduce = False

        if not isinstance(tokens, list):
            tokens = [tokens]
            to_reduce = True

        hit = 0
        cnt = 0
        indices = []
        if not lower_case_backup:
            for token in tokens:
                result = self[token]
                indices.append(result[0])
                hit += int(result[1])
                cnt += 1
        else:
            for token in tokens:
                result = self[token] if token in self.stoi else self[token.lower()]
                indices.append(result[0])
                hit += int(result[1])
                cnt += 1

        vecs = torch.stack(indices)
        return vecs[0] if to_reduce else vecs, hit, cnt


class Vocab(object):
    """Vocab class.

    Parameters
    ----------
    lower_case : boolean
        Specify whether to lowercase text, default: ``True``.
    tokenizer: function, optional
        Word tokenization function, default: nltk.tokenize.word_tokenize.

    Examples
    -------
    >>> word_vocab = Vocab()
    >>> word_vocab.build_vocab({'i': 10, 'like': 5, 'nlp': 3})
    >>> print(word_vocab.get_vocab_size())
    """

    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3
    pad_token = constants._PAD_TOKEN
    sos_token = constants._SOS_TOKEN
    eos_token = constants._EOS_TOKEN
    unk_token = constants._UNK_TOKEN

    def __init__(self, lower_case=True, tokenizer=word_tokenize):
        super(Vocab, self).__init__()
        self.lower_case = lower_case
        self.tokenizer = tokenizer
        self.reserved = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.index2word = self.reserved[:]
        self.word2index = dict(zip(self.reserved, range(len(self.reserved))))
        self.word2count = Counter()
        self.embeddings = None

    def build_vocab(self, vocab_counter, max_vocab_size=None, min_vocab_freq=1):
        """Build vocab from ``vocab_counter`` which is a vocab count dict.

        Parameters
        ----------
        vocab_counter : dict
            Vocab counter.
        max_vocab_size : None, optional
            Maximal word vocab size, default: ``None``.
        min_vocab_freq : int, optional
            Minimal word vocab frequency, default: ``1``.
        """
        self.word2count = vocab_counter
        self._add_words(vocab_counter.keys())
        self._trim(max_vocab_size=max_vocab_size, min_vocab_freq=min_vocab_freq)

    def _add_words(self, words):
        # words: a list of str
        for word in words:
            if self.lower_case:
                word = word.lower()

            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        assert len(self.word2index) == len(self.index2word)

    def _trim(self, max_vocab_size=None, min_vocab_freq=1):
        """Trim vocab"""
        if min_vocab_freq <= 1 and (
            max_vocab_size is None or max_vocab_size >= len(self.word2index)
        ):
            return

        ordered_words = sorted(((c, w) for (w, c) in self.word2count.items()), reverse=True)

        if max_vocab_size:
            ordered_words = ordered_words[:max_vocab_size]

        self.index2word = self.reserved[:]
        self.word2index = dict(zip(self.reserved, range(len(self.reserved))))
        self.word2count = Counter()

        for count, word in ordered_words:
            if count < min_vocab_freq:
                break
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.word2count[word] = count
                self.index2word.append(word)

        assert len(self.word2index) == len(self.index2word)

    def load_embeddings(
        self,
        pretrained_word_emb_name="840B",
        pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=".vector_cache/",
        pretrained_word_emb_dim=300,
        dtype=np.float32,
    ):
        """Load pretrained word embeddings for initialization"""
        word_model = WordEmbModel(
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            pretrained_word_emb_dim=pretrained_word_emb_dim,
        )

        word_list = list(self.word2index.keys())

        word_emb, hit, cnt = word_model.get_vecs_by_tokens(
            tokens=word_list, lower_case_backup=self.lower_case
        )

        print("Pretrained word embeddings hit ratio: {}".format(hit / cnt))
        self.embeddings = word_emb.numpy()
        self.embeddings[self.PAD] = np.zeros(word_model.dim)

    def randomize_embeddings(self, n_dims, scale=0.08):
        """Use random word embeddings for initialization."""
        vocab_size = self.get_vocab_size()
        shape = (vocab_size, n_dims)
        self.embeddings = np.array(
            np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32
        )
        self.embeddings[self.PAD] = np.zeros(n_dims)

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    @lru_cache(maxsize=None)
    def is_word(self, token_id: int) -> bool:
        """Return whether the token at `token_id` is a word; False for punctuations."""
        if token_id < 4:
            return False
        if token_id >= len(self):
            return True  # OOV is assumed to be words
        token_str = self.index2word[token_id]
        if not word_detector.search(token_str) or token_str == "<P>":
            return False
        return True

    def get_vocab_size(self):
        return len(self.index2word)

    def getIndex(self, word, use_ie=False):
        # For IE Graph
        # word can be a phrase
        if use_ie:
            if self.lower_case:
                word = word.lower()

            ret = []
            for x in word.replace("_", " ").split(" "):
                if x == "":
                    continue
                ret.append(self.word2index.get(x, self.UNK))

            return ret

        if self.lower_case:
            word = word.lower()

        return self.word2index.get(word, self.UNK)

    def getWord(self, idx):
        return self.index2word[idx] if idx < len(self.index2word) else self.unk_token

    def to_word_sequence(self, seq):
        sentence = []
        for idx in seq:
            word = self.getWord(idx)
            sentence.append(word)
        return " ".join(sentence)

    def to_index_sequence(self, sentence):
        sentence = sentence.strip()
        if self.lower_case:
            sentence = sentence.lower()

        seq = []
        if self.tokenizer is None:
            for word in sentence.split():
                idx = self.getIndex(word)
                seq.append(idx)
        else:
            for word in self.tokenizer(sentence):
                idx = self.getIndex(word)
                seq.append(idx)
        return seq

    def to_index_sequence_for_list(self, words):
        seq = []
        for word in words:
            if self.lower_case:
                word = word.lower()

            idx = self.getIndex(word)
            seq.append(idx)
        return seq
