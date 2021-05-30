"""
graph4nlp.datasets module contains various common datasets implemented based on graph4nlp.data.dataset.
"""

from .geo import GeoDatasetForTree
from .jobs import JobsDataset, JobsDatasetForTree
from .mawps import MawpsDatasetForTree
from .squad import SQuADDataset
from .trec import TrecDataset
from .mathqa import MathQADatasetForTree