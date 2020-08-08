import abc
import copy
import os

import torch.utils.data

from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from ..modules.utils.vocab_utils import VocabModel


class Dataset(torch.utils.data.Dataset):
    """
    Base class for datasets.

    Parameters
    ----------
    root: str
        The root directory path where the dataset is stored.
    """

    @property
    def raw_file_names(self) -> list:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> list:
        raise NotImplementedError

    @property
    def topology_subdir(self):
        raise NotImplementedError

    @abc.abstractmethod
    def download(self):
        """Download the raw data from the Internet."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_topology(self):
        raise NotImplementedError

    @abc.abstractmethod
    def build_vocab(self):
        raise NotImplementedError

    @abc.abstractmethod
    def vectorization(self):
        raise NotImplementedError

    def len(self):
        """The number of examples in the whole dataset."""
        raise NotImplementedError

    def get(self, index: int):
        """The abstraction of getting one item from the dataset given the index."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def collate_fn(data_list):
        """Takes a list of data and convert it to a batch of data."""
        raise NotImplementedError

    def __init__(self, root, topology_builder, vocab_builder=VocabModel):
        super(Dataset, self).__init__()

        self.root = root
        self.topology_builder = topology_builder
        self.vocab_builder = vocab_builder
        self.__indices__ = None

        if 'download' in self.__class__.__dict__.keys():
            self._download()

        self._process()

    @property
    def raw_dir(self) -> str:
        """The directory where the raw data is stored."""
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed', self.topology_subdir)

    @property
    def raw_file_paths(self) -> list:
        """The paths to raw files."""
        return [os.path.join(self.raw_dir, raw_file_name) for raw_file_name in self.raw_file_names]

    @property
    def processed_file_paths(self) -> list:
        return [os.path.join(self.processed_dir, processed_file_name) for processed_file_name in
                self.processed_file_names]

    def _download(self):
        if all([os.path.exists(raw_path) for raw_path in self.raw_file_paths]):
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def _process(self):
        if all([os.path.exists(processed_path) for processed_path in self.processed_file_paths]):
            return

        os.makedirs(self.processed_dir, exist_ok=True)

        if 'build_topology' in self.__class__.__dict__.keys():
            self.build_topology()
        if 'build_vocab' in self.__class__.__dict__.keys():
            self.build_vocab()
        if 'vectorization' in self.__class__.__dict__.keys():
            self.vectorization()

    def indices(self):
        if self.__indices__ is not None:
            return self.__indices__
        else:
            return range(len(self))

    def index_select(self, idx):
        indices = self.indices()

        if isinstance(idx, slice):
            indices = indices[idx]
        elif torch.is_tensor(idx):
            if idx.dtype == torch.long:
                if len(idx.shape) == 0:
                    idx = idx.unsqueeze(0)
                return self.index_select(idx.tolist())
            elif idx.dtype == torch.bool or idx.dtype == torch.uint8:
                return self.index_select(idx.nonzero().flatten().tolist())
        elif isinstance(idx, list) or isinstance(idx, tuple):
            indices = [indices[i] for i in idx]
        else:
            raise IndexError(
                'Only integers, slices (`:`), list, tuples, and long or bool '
                'tensors are valid indices (got {}).'.format(
                    type(idx).__name__))

        dataset = copy.copy(self)
        dataset.__indices__ = indices
        return dataset

    def shuffle(self, return_perm=False):
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def __len__(self):
        if self.__indices__ is not None:
            return len(self.__indices__)
        else:
            return self.len()

    def __getitem__(self, index):
        if not isinstance(index, int):
            return self.index_select(index)
        else:
            return self.get(self.indices()[index])


class DependencyDataset(Dataset):
    @property
    def topology_subdir(self):
        return os.path.join(self.root, 'processed', 'DependencyGraph')

    def __init__(self, root):
        super(DependencyDataset, self).__init__(root=root, topology_builder=DependencyBasedGraphConstruction)
