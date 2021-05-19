<p align="center">
<img src="./imgs/logo_graph4nlp.png" width="500" class="center" alt="logo"/>
    <br/>
</p>

[pypi-image]: https://badge.fury.io/py/graph4nlp.svg

[pypi-url]: https://pypi.org/project/graph4nlp

[license-image]:https://img.shields.io/badge/License-Apache%202.0-blue.svg

[license-url]:https://github.com/hugochan/IDGL/blob/master/LICENSE

[contributor-image]:https://img.shields.io/github/contributors/hugochan/IDGL

[contributor-url]:https://github.com/hugochan/IDGL/contributors

[contributing-image]:https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat

[contributing-url]:to_be_add

[issues-image]:https://img.shields.io/github/issues/hugochan/IDGL

[issues-url]:https://github.com/hugochan/IDGL/issues

[forks-image]:https://img.shields.io/github/forks/hugochan/IDGL?style=social

[forks-url]:https://github.com/hugochan/IDGL/fork

[stars-image]:https://img.shields.io/github/stars/hugochan/IDGL?style=social

[stars-url]:https://github.com/hugochan/IDGL/star

### for repo

![Last Commit](https://img.shields.io/github/last-commit/hugochan/IDGL)
[![pypi][pypi-image]][pypi-url]
[![Contributors][contributor-image]][contributor-url]
[![Contributing][contributing-image]][contributing-url]
[![License][license-image]][license-url]
[![Issues][issues-image]][issues-url]

### for docs

[![Fork][forks-image]][forks-url]
[![Star][stars-image]][stars-url]

# Graph4NLP

***Graph4NLP*** is an easy-to-use library for R&D at the intersection of **Graph Deep Learning** and
**Natural Language Processing**. It provides both **full implementations** of state-of-the-art models for data scientists and also **flexible interfaces** to build custom models for researchers and developers with whole-pipeline support. Built upon highly-optimized runtime libraries including [DGL](https://github.com/dmlc/dgl) , ***Graph4NLP*** is both high running effieciency and great extensibility. The architecture of ***Graph4NLP*** is shown in the following figure, where boxes with dashed lines represents the features under development.

<img src="docs/arch.png" alt="architecture" style="zoom:25%;" />

## <img src="docs/new.png" alt='new' width=30 /> Graph4NLP news
**20/05/2021:** The **v1.0.0 release**. Try it out!

## Quick tour

***Graph4nlp*** aims to make it incredibly easy to use GNNs in NLP tasks (
see [here](http://saizhuo.wang/g4nlp/index.html) for tutorial). Here is a example of how to use the [*
Graph2seq*](http://saizhuo.wang/g4nlp/index.html) model (widely used in machine translation, question answering,
semantic parsing, and various other nlp tasks that can be abstracted as graph to sequence problem and show superior
performance).

If you want to further improve model performance, we also provide interfaces of pre-trained models,
including [GloVe](https://nlp.stanford.edu/pubs/glove.pdf), [BERT](https://arxiv.org/abs/1810.04805), etc.

We also offer many other effective models such as graph classification models, graph to tree models, etc. If you are
interested in related research problems, welcome to use our library and refer to our [graph4nlp survey](to_be_add).

```python
from graph4nlp.pytorch.datasets.jobs import JobsDataset
from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.modules.utils.config_utils import update_values, get_yaml_config

# build dataset
jobs_dataset = JobsDataset(root_dir='graph4nlp/pytorch/test/dataset/jobs',
                           topology_builder=DependencyBasedGraphConstruction,
                           topology_subdir='DependencyGraph')  # You should run stanfordcorenlp at background
vocab_model = jobs_dataset.vocab_model

# build model
user_args = get_yaml_config("examples/pytorch/semantic_parsing/graph2seq/config/dependency_gcn_bi_sep_demo.yaml")
args = get_basic_args(graph_construction_name="node_emb", graph_embedding_name="gat", decoder_name="stdrnn")
update_values(to_args=args, from_args_list=[user_args])
graph2seq = Graph2Seq.from_args(args, vocab_model)

# calculation
batch_data = JobsDataset.collate_fn(jobs_dataset.train[0:12])

scores = graph2seq(batch_data["graph_data"], batch_data["tgt_seq"])  # [Batch_size, seq_len, Vocab_size]
```

## Overview

<p align="center">
<img src="./imgs/graph4nlp_flow_v2.png" width="1000" class="center" alt="logo"/>
    <br/>
</p>

## Graph4NLP Models and Applications

Graph2Seq, Seq2Seq, Graph2Tree

## Installation

Currently, users can install Graph4NLP via **pip** or **source code**. Graph4NLP supports the following OSes:

- Linux-based systems (tested on Ubuntu 18.04 and later)
- macOS (only CPU version)
- Windows 10 

### Installation via pip

| Platform  | Command                       |
| --------- | ----------------------------- |
| CPU       | `pip install graph4nlp`       |
| CUDA 9.2  | `pip install graph4nlp-cu92`  |
| CUDA 10.1 | `pip install graph4nlp-cu101` |
| CUDA 10.2 | `pip install graph4nlp-cu102` |
| CUDA 11.0 | `pip install graph4nlp-cu110` |

### Installation via source code

The source code of Graph4NLP can be retrieved from GitHub:

```bash
git clone https://github.com/graph4ai/graph4nlp.git
cd graph4nlp
```

Then run `./configure` (or `./configure.bat`  if you are using Windows 10) to config your installation. The configuration program will ask you to specify your CUDA version. If you do not have a GPU, please type 'cpu'.

```bash
./configure
```

Finally, use pip to install the package:

```shell
python setup.py install
```

## Major Releases

| Releases | Date       | Features                                                     |
| -------- | ---------- | ------------------------------------------------------------ |
| v1.0.0   | 2021-05-20 | - Support the whole pipeline of Graph4NLP<br />- GraphData and Dataset support |

## More resources

If you are new to using graph deep learning methods for natural language processing tasks, you can refer to our survey paper which provides an overview of this research direction. If you want detailed reference to  our library, please refer to our docs.

[Docs]() | [Graph4nlp survey]() | [Related paper list]() | [Workshops]()

## Contributing

Please let us know if you encounter a bug or have any suggestions by filing an issue.

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs. 

## Citation

If you found this code useful, please consider citing the following paper:

Yu Chen, Lingfei Wu and Mohammed J. Zaki. **"Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust
Node Embeddings."** In *Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Dec
6-12, 2020.*

    @article{chen2020iterative,
      title={Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings},
      author={Chen, Yu and Wu, Lingfei and Zaki, Mohammed},
      journal={Advances in Neural Information Processing Systems},
      volume={33},
      year={2020}
    }

## Team
Graph4AI Team. Lingfei Wu, Yu Chen, Kai Shen, Hanning Gao, Xiaojie Guo, Shucheng Li, Saizhuo Wang

## License
Graph4NLP uses Apache License 2.0.