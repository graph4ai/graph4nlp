import abc
import os

import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    """
    Dataset base class for creating datasets from raw data.

    Parameters
    ----------
    root: str
        The root directory path where the dataset is stored.
    """

    @property
    def raw_file_names(self) -> list:
        """The file names of raw data which are to be found in the `self.raw_dir` directory.

        Returns
        -------
        list
            The list of raw file names.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def download(self):
        """Download the raw data from the Internet."""
        raise NotImplementedError

    def __init__(self, root):
        super(Dataset, self).__init__()

        self.root = root
        self.__indices__ = None

        if 'download' in self.__class__.__dict__.keys():
            self._download()

    @property
    def raw_dir(self) -> str:
        """The directory where the raw data is stored."""
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_paths(self) -> list:
        """The paths to raw files."""
        return [os.path.join(self.raw_dir, raw_file_name) for raw_file_name in self.raw_file_names]

    def _download(self):
        if all([os.path.exists(raw_path) for raw_path in self.raw_file_paths]):
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def len(self):
        """The number of examples in the whole dataset."""
        raise NotImplementedError

    def get(self, index):
        """The abstraction of getting one item from the dataset given the index."""
        raise NotImplementedError

    def __len__(self):
        if self.__indices__ is not None:
            return len(self.__indices__)
        else:
            return self.len()

    # TODO: Design the `self.indices` method.

    # TODO: Vectorization of data (step 3 of preprocessing). Runtime or static?

    def __getitem__(self, index):
        # TODO: Implement __getitem__ method.
        pass

    @staticmethod
    @abc.abstractmethod
    def collate_fn(self):
        """Takes a list of data and convert it to a batch of data."""
        raise NotImplementedError


class TopologyDataset(Dataset):
    """
    The base class for dataset with topology building techniques.
    """

    @property
    def topology_file_names(self):
        """The file names of the topological data."""
        raise NotImplementedError

    @property
    def vocab_file_names(self):
        """The file names of the vocabulary."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_topology(self):
        """Build topology based on raw data."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_vocab(self):
        """Build vocabulary based on the topology."""
        raise NotImplementedError

    def __init__(self, root, topology_build_fn, vocab_build_fn):
        super(TopologyDataset, self).__init__(root=root)

        self.vocab_build_fn = vocab_build_fn
        self.topology_build_fn = topology_build_fn

        if 'build_topology' in self.__class__.__dict__.keys():
            self._build_topology()

        if 'build_vocab' in self.__class__.__dict__.keys():
            self._build_vocab()

    @property
    def topology_dir(self):
        return os.path.join(self.root, 'topology')

    @property
    def vocab_dir(self):
        return os.path.join(self.root, 'vocab')

    @property
    def topology_file_paths(self):
        return [os.path.join(self.topology_dir, file_name) for file_name in self.topology_file_names]

    @property
    def vocab_file_paths(self):
        return [os.path.join(self.vocab_dir, file_name) for file_name in self.vocab_file_names]

    def _build_topology(self):
        if all([os.path.exists(file_path) for file_path in self.topology_file_paths]):
            return

        os.makedirs(self.topology_dir, exist_ok=True)
        self.build_topology()

    def _build_vocab(self):
        if all([os.path.exists(file_path) for file_path in self.vocab_file_paths]):
            return

        os.makedirs(self.vocab_dir, exist_ok=True)
        self._build_vocab()

class TopologyBuilder(object):
    """
    The base class for dataset with topology building techniques.
    """

    @property
    def topology_file_names(self):
        """The file names of the topological data."""
        raise NotImplementedError

    @property
    def vocab_file_names(self):
        """The file names of the vocabulary."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_topology(self):
        """Build topology based on raw data."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_vocab(self):
        """Build vocabulary based on the topology."""
        raise NotImplementedError

    def __init__(self, topology_build_fn, vocab_build_fn):
        super(TopologyBuilder, self).__init__()

        self.vocab_build_fn = vocab_build_fn
        self.topology_build_fn = topology_build_fn

        if 'build_topology' in self.__class__.__dict__.keys():
            self._build_topology()

        if 'build_vocab' in self.__class__.__dict__.keys():
            self._build_vocab()

    @property
    def topology_dir(self):
        return os.path.join(self.root, 'topology')

    @property
    def vocab_dir(self):
        return os.path.join(self.root, 'vocab')

    @property
    def topology_file_paths(self):
        return [os.path.join(self.topology_dir, file_name) for file_name in self.topology_file_names]

    @property
    def vocab_file_paths(self):
        return [os.path.join(self.vocab_dir, file_name) for file_name in self.vocab_file_names]

    def _build_topology(self):
        if all([os.path.exists(file_path) for file_path in self.topology_file_paths]):
            return

        os.makedirs(self.topology_dir, exist_ok=True)
        self.build_topology()

    def _build_vocab(self):
        if all([os.path.exists(file_path) for file_path in self.vocab_file_paths]):
            return

        os.makedirs(self.vocab_dir, exist_ok=True)
        self._build_vocab()
