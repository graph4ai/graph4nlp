"""
graph4nlp.datasets module contains various common datasets implemented based on graph4nlp.data.dataset.
"""

from .cnn import CNNDataset
from .geo import GeoDatasetForTree
from .jobs import JobsDataset, JobsDatasetForTree
from .kinship import KinshipDataset
from .mawps import MawpsDatasetForTree
from .squad import SQuADDataset
from .trec import TrecDataset
from .wn18rr import WN18RRDataset