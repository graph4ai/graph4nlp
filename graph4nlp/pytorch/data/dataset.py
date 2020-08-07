import abc
import os

import torch.utils.data

from ..modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from ..modules.graph_construction.base import GraphConstructionBase
from ..modules.utils.vocab_utils import VocabModel


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
        raise NotImplementedError

    @property
    def processed_file_names(self) -> list:
        raise NotImplementedError

    @property
    def preprocessed_file_names(self) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    def download(self):
        """Download the raw data from the Internet."""
        raise NotImplementedError

    @abc.abstractmethod
    def vectorization(self):
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self):
        raise NotImplementedError

    def __init__(self, root):
        super(Dataset, self).__init__()

        self.root = root
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
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_paths(self) -> list:
        """The paths to raw files."""
        return [os.path.join(self.raw_dir, raw_file_name) for raw_file_name in self.raw_file_names]

    @property
    def processed_file_paths(self) -> list:
        return [os.path.join(self.processed_dir, processed_file_name) for processed_file_name in
                self.processed_file_names]

    @property
    def preprocessed_file_paths(self):
        return [os.path.join(self.processed_dir, file_name) for file_name in self.preprocessed_file_names]

    def _download(self):
        if all([os.path.exists(raw_path) for raw_path in self.raw_file_paths]):
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    def _process(self):
        if all([os.path.exists(processed_path) for processed_path in self.processed_file_paths]):
            return

        os.makedirs(self.processed_dir, exist_ok=True)
        print("iiiiii", self.__class__.__dict__)

        if 'preprocess' in self.__class__.__dict__.keys():
            self._preprocess()
        if 'build_topology' in self.__class__.__dict__.keys():
            self._build_topology()
        if 'build_vocab' in self.__class__.__dict__.keys():
            self._build_vocab()
        self.vectorization()

    def _preprocess(self):
        # if all([os.path.exists(preprocessed_path) for preprocessed_path in self.preprocessed_file_paths]):
        #     self.seq_data = torch.load(os.path.join(self.processed_dir, 'sequence.pt'))
        #     return

        os.makedirs(self.processed_dir, exist_ok=True)
        self.preprocess()

    def len(self):
        """The number of examples in the whole dataset."""
        raise NotImplementedError

    def get(self, index: int):
        """The abstraction of getting one item from the dataset given the index."""
        raise NotImplementedError

    def __len__(self):
        if self.__indices__ is not None:
            return len(self.__indices__)
        else:
            return self.len()

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise NotImplementedError('Currently only index of int type is supported in `Dataset`.')
        else:
            return self.get(index)

    @staticmethod
    @abc.abstractmethod
    def collate_fn(data_list):
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

    @property
    def topology_subdir_name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def build_topology(self):
        """Build topology based on raw data."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_vocab(self):
        """Build vocabulary based on the topology."""
        raise NotImplementedError

    def __init__(self, root, topology_builder: GraphConstructionBase, vocab_builder=VocabModel):
        self.vocab_builder = vocab_builder
        self.topology_builder = topology_builder
        super(TopologyDataset, self).__init__(root=root)

    @property
    def topology_subdir(self):
        return os.path.join(self.root, 'processed', self.topology_subdir_name)

    @property
    def topology_file_paths(self):
        return [os.path.join(self.topology_subdir, file_name) for file_name in self.topology_file_names]

    @property
    def vocab_file_paths(self):
        return [os.path.join(self.topology_subdir, file_name) for file_name in self.vocab_file_names]

    def _build_topology(self):
        if all([os.path.exists(file_path) for file_path in self.topology_file_paths]):
            self.topo_data = torch.load(os.path.join(self.topology_subdir, 'topology.pt'))
            return

        os.makedirs(self.topology_subdir, exist_ok=True)
        self.build_topology()

    def _build_vocab(self):
        # if all([os.path.exists(file_path) for file_path in self.vocab_file_paths]):
        #     return
        os.makedirs(self.topology_subdir, exist_ok=True)
        self.build_vocab()


class DependencyDataset(TopologyDataset):
    @property
    def topology_subdir_name(self):
        return 'DependencyGraph'

    def __init__(self, root):
        super(DependencyDataset, self).__init__(root=root, topology_builder=DependencyBasedGraphConstruction)
