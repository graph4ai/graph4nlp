import os
import pickle
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy import sparse
from sklearn import preprocessing

from graph4nlp.pytorch.data.data import GraphData


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.safe_load(setting)
    return config


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns
        always a (possibly empty) dictionary
        """
        from functools import reduce

        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [
        merge_dicts(
            {k: v for k, v in kwargs.items() if not isinstance(v, list)},
            {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)},
        )
        for vv in grd
    ]


def to_cuda(x, device=None):
    if device:
        x = x.to(device)

    return x


def all_to_cuda(data, device=None):
    if isinstance(data, torch.Tensor) or isinstance(data, GraphData):
        data = to_cuda(data, device)
    elif isinstance(data, (list, dict)):
        keys = range(len(data)) if isinstance(data, list) else data.keys()
        for k in keys:
            if isinstance(data[k], torch.Tensor) or isinstance(data[k], GraphData):
                data[k] = to_cuda(data[k], device)
    return data


def create_mask(x, N, device=None):
    if isinstance(x, torch.Tensor):
        x = x.data
    mask = np.zeros((len(x), N))
    for i in range(len(x)):
        mask[i, : x[i]] = 1

    return to_cuda(torch.Tensor(mask), device)


def normalize_adj(mx):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)

    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


# def normalize_adj(mx):
#     """Normalize matrix: asymmetric normalized Laplacian"""
#     rowsum = mx.sum(1)
#     r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
#     r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
#     r_mat_inv_sqrt = torch.diag(r_inv_sqrt)

#     colsum = mx.sum(0)
#     c_inv_sqrt = torch.pow(colsum, -0.5).flatten()
#     c_inv_sqrt[torch.isinf(c_inv_sqrt)] = 0.
#     c_mat_inv_sqrt = torch.diag(c_inv_sqrt)

#     return torch.mm(torch.mm(r_mat_inv_sqrt, mx), c_mat_inv_sqrt)


def normalize_sparse_adj(mx):
    """symmetric normalized Laplacian"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sparse.diags(r_inv_sqrt)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def dropout_fn(x, drop_prob, shared_axes=None, training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    x: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
        with dropout applied.
    """
    if drop_prob == 0 or drop_prob is None or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes or []:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1.0 - drop_prob).div_(1.0 - drop_prob)
    mask = mask.expand_as(x)

    return x * mask


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class EarlyStopping:
    def __init__(self, save_model_path, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_model_path = save_model_path

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            save_model_checkpoint(model, self.save_model_path)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_model_checkpoint(model, self.save_model_path)
            self.counter = 0

        return self.early_stop


def save_model_checkpoint(model, model_path):
    """The API for saving the model.

    Parameters
    ----------
    model : class
        The model to be saved.
    model_path : str
        The saved model path.

    Returns
    -------
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)
    print("Saved model to {}".format(model_path))


class LabelModel(object):
    """Class for building label mappgings from a label set."""

    def __init__(self, all_labels=None):
        super(LabelModel, self).__init__()
        self.le = preprocessing.LabelEncoder()
        self.le.fit(list(all_labels))
        self.num_classes = len(self.le.classes_)

    @classmethod
    def build(cls, saved_label_file, all_labels=None):
        if os.path.exists(saved_label_file):
            print("Loading pre-built label mappings stored in {}".format(saved_label_file))
            with open(saved_label_file, "rb") as f:
                label_model = pickle.load(f)
        else:
            label_model = cls(all_labels)
            print("Saving label mappings to {}".format(saved_label_file))
            pickle.dump(label_model, open(saved_label_file, "wb"))

        return label_model


def wordid2str(word_ids, vocab):
    ret = []
    assert len(word_ids.shape) == 2, print(word_ids.shape)
    for i in range(word_ids.shape[0]):
        id_list = word_ids[i, :]
        ret_inst = []
        for j in range(id_list.shape[0]):
            if id_list[j] == vocab.EOS or id_list[j] == vocab.PAD:
                break
            token = vocab.getWord(id_list[j])
            ret_inst.append(token)
        ret.append(" ".join(ret_inst))
    return ret
