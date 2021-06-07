<p align="center">
<img src="./imgs/graph4nlp_logo.png" width="800" class="center" alt="logo"/>
    <br/>
</p>
 
[pypi-image]: https://badge.fury.io/py/graph4nlp.svg

[pypi-url]: https://pypi.org/project/graph4nlp

[license-image]:https://img.shields.io/badge/License-Apache%202.0-blue.svg

[license-url]:https://github.com/graph4ai/graph4nlp/blob/master/LICENSE

[contributor-image]:https://img.shields.io/github/contributors/graph4ai/graph4nlp

[contributor-url]:https://github.com/graph4ai/graph4nlp/contributors

[contributing-image]:https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat

[contributing-url]:to_be_add

[issues-image]:https://img.shields.io/github/issues/graph4ai/graph4nlp

[issues-url]:https://github.com/graph4ai/graph4nlp/issues

[forks-image]:https://img.shields.io/github/forks/graph4ai/graph4nlp?style=social

[forks-url]:https://github.com/graph4ai/graph4nlp/fork

<!-- [stars-image]:https://img.shields.io/github/stars/graph4ai/graph4nlp?style=social

[stars-url]:https://github.com/graph4ai/graph4nlp/stars -->

![Last Commit](https://img.shields.io/github/last-commit/graph4ai/graph4nlp)
[![pypi][pypi-image]][pypi-url]
[![Contributors][contributor-image]][contributor-url]
[![Contributing][contributing-image]][contributing-url]
[![License][license-image]][license-url]
[![Issues][issues-image]][issues-url]

<!-- ### for docs

[![Fork][forks-image]][forks-url]
[![Star][stars-image]][stars-url] -->

# Graph4NLP

***Graph4NLP*** is an easy-to-use library for R&D at the intersection of **Deep Learning on Graphs** and
**Natural Language Processing** (i.e., DLG4NLP). It provides both **full implementations** of state-of-the-art models for data scientists and also **flexible interfaces** to build customized models for researchers and developers with whole-pipeline support. Built upon highly-optimized runtime libraries including [DGL](https://github.com/dmlc/dgl) , ***Graph4NLP*** has both high running efficiency and great extensibility. The architecture of ***Graph4NLP*** is shown in the following figure, where boxes with dashed lines represents the features under development. Graph4NLP consists of four different layers: 1) Data Layer, 2) Module Layer, 3) Model Layer, and 4) Application Layer.

<p align="center">
    <img src="docs/arch.png" alt="architecture" width="700" />
    <br>
    <b>Figure</b>: Graph4NLP Overall Architecture
</p>

## <img src="docs/new.png" alt='new' width=30 /> Graph4NLP news
**06/05/2021:** The **v0.4.1 release**. Try it out!

## Quick tour

***Graph4nlp*** aims to make it incredibly easy to use GNNs in NLP tasks (check out [Graph4NLP Documentation](http://saizhuo.wang/g4nlp/index.html)). Here is an example of how to use the [*Graph2seq*](http://saizhuo.wang/g4nlp/index.html) model (widely used in machine translation, question answering,
semantic parsing, and various other NLP tasks that can be abstracted as graph-to-sequence problem and has shown superior
performance).

<!-- If you want to further improve model performance, we also support pre-trained models including [BERT](https://arxiv.org/abs/1810.04805), etc.
 -->
We also offer other high-level model APIs such as graph-to-tree models. If you are interested in DLG4NLP related research problems, you are very welcome to use our library and refer to our [graph4nlp survey](to_be_add).

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

Our Graph4NLP computing flow is shown as below.
<p align="center">
<img src="./imgs/graph4nlp_flow.png" width="1000" class="center" alt="logo"/>
    <br/>
</p>

## Graph4NLP Models and Applications

### Graph4NLP models

- [Graph2Seq](https://github.com/graph4ai/graph4nlp/blob/master/graph4nlp/pytorch/models/graph2seq.py): a general end-to-end neural encoder-decoder model that maps an input graph to a sequence of tokens.  
- [Graph2Tree](https://github.com/graph4ai/graph4nlp/blob/master/graph4nlp/pytorch/models/graph2tree.py): a general end-to-end neural encoder-decoder model that maps an input graph to a tree structure.

### Graph4NLP applications

We provide a comprehensive collection of NLP applications, together with detailed examples as follows:

- [Text classification](https://github.com/graph4ai/graph4nlp/tree/master/examples/pytorch/text_classification): to give the sentence or document an appropriate label.
- [Semantic parsing](https://github.com/graph4ai/graph4nlp/tree/master/examples/pytorch/semantic_parsing): to translate natural language into a machine-interpretable formal meaning representation.
- [Neural machine translation](https://github.com/graph4ai/graph4nlp/tree/master/examples/pytorch/nmt): to translate a sentence in a source language to a different target language.
- [summarization](https://github.com/graph4ai/graph4nlp/tree/master/examples/pytorch/summarization): to generate a shorter version of input texts which could preserve major meaning.
- [KG completion](https://github.com/graph4ai/graph4nlp/tree/master/examples/pytorch/kg_completion): to predict missing relations between two existing entities in konwledge graphs.
- [Math word problem solving](https://github.com/graph4ai/graph4nlp/tree/master/examples/pytorch/math_word_problem): to automatically solve mathematical exercises that provide background information about a problem in easy-to-understand language.
- [Name entity recognition](https://github.com/graph4ai/graph4nlp/tree/master/examples/pytorch/name_entity_recognition): to tag entities in input texts with their corresponding type.
- [Question generation](https://github.com/graph4ai/graph4nlp/tree/master/examples/pytorch/question_generation): to generate an valid and fluent question based on the given passage and target answer (optional).


## Performance

| Task                       |              Dataset             |   GNN    Model      | Graph construction                           | Evaluation         |          Performance          |
|----------------------------|:--------------------------------:|:-------------------:|----------------------------------------------|--------------------|:-----------------------------:|
| Text classification        | TRECT<br> CAirline<br> CNSST<br> |           GAT       | Dependency                                   |        Accuracy    | 0.948<br> 0.769<br> 0.538<br> |
| Semantic Parsing           |               JOBS               |           SAGE      | Constituency                                 | Execution accuracy |             0.936             |
| Question generation        |               SQuAD             |           GGNN       | Dependency                                      | BLEU-4             |             0.15175	            |
| Machine translation        |              IWSLT14             |           GCN       | Dynamic                                      | BLEU-4             |             0.3212            |
| Summarization              |             CNN(30k)             |           GCN       | Dependency                                   | ROUGE-1            |              26.4             |
| Knowledge graph completion | Kinship                          |           GCN      | Dependency                                    | MRR                | 82.4                          |
| Math word problem          | MAWPS  <br> MATHQA               | SAGE                | Dynamic                                      | Solution accuracy <br> Exact match  | 76.4<br>  61.07  |

## Installation

Currently, users can install Graph4NLP via **pip** or **source code**. Graph4NLP supports the following OSes:

- Linux-based systems (tested on Ubuntu 18.04 and later)
- macOS (only CPU version)
- Windows 10 (only support pytorch >= 1.8)

### Installation via pip (binaries)
We provide pip wheels for all major OS/PyTorch/CUDA combinations. Note that we highly recommend `Windows` users refer to `Installation via source code` due to compatibility.

#### Ensure that at least PyTorch (>=1.6.0) is installed:
Note that `>=1.6.0` is ok.
``` bash
$ python -c "import torch; print(torch.__version__)"
>>> 1.6.0
```
#### Find the CUDA version PyTorch was installed with (for GPU users):
```bash
$ python -c "import torch; print(torch.version.cuda)"
>>> 10.2
```

#### Install the relevant dependencies:
`torchtext` is needed since Graph4NLP relies on it to implement embeddings.
Please pay attention to the PyTorch requirements before installing `torchtext` with the following script! For detailed version matching please refer [here](https://pypi.org/project/torchtext/).
``` bash
pip install torchtext # >=0.7.0
```


#### Install Graph4NLP
```bash
pip install graph4nlp${CUDA}
```
where `${CUDA}` should be replaced by the specific CUDA version (`none` (CPU version), `"-cu92"`, `"-cu101"`, `"-cu102"`, `"-cu110"`). The following table shows the concrete command lines. For CUDA 11.1 users, please refer to `Installation via source code`.

| Platform  | Command                       |
| --------- | ----------------------------- |
| CPU       | `pip install graph4nlp`   |
| CUDA 9.2  | `pip install graph4nlp-cu92`  |
| CUDA 10.1 | `pip install graph4nlp-cu101` |
| CUDA 10.2 | `pip install graph4nlp-cu102` |
| CUDA 11.0 | `pip install graph4nlp-cu110` |

### Installation via source code

#### Ensure that at least PyTorch (>=1.6.0) is installed:
Note that `>=1.6.0` is ok.
``` bash
$ python -c "import torch; print(torch.__version__)"
>>> 1.6.0
```
#### Find the CUDA version PyTorch was installed with (for GPU users):
```bash
$ python -c "import torch; print(torch.version.cuda)"
>>> 10.2
```

#### Install the relevant dependencies:
`torchtext` is needed since Graph4NLP relies on it to implement embeddings.
Please pay attention to the PyTorch requirements before installing `torchtext` with the following script! For detailed version matching please refer [here](https://pypi.org/project/torchtext/).
``` bash
pip install torchtext # >=0.7.0
```

#### Download the source code of `Graph4NLP` from Github:
```bash
git clone https://github.com/graph4ai/graph4nlp.git
cd graph4nlp
```
#### Configure the CUDA version
Then run `./configure` (or `./configure.bat`  if you are using Windows 10) to config your installation. The configuration program will ask you to specify your CUDA version. If you do not have a GPU, please type 'cpu'.
```bash
./configure
```

#### Install the relevant packages:

Finally, install the package:

```shell
python setup.py install
```

## Major Releases

| Releases | Date       | Features                                                     |
| -------- | ---------- | ------------------------------------------------------------ |
| v0.4.1   | 2021-06-05 | - Support the whole pipeline of Graph4NLP<br />- GraphData and Dataset support |

## New to Deep Learning on Graphs for NLP?

If you want to learn more on applying Deep Learning on Graphs techniques to NLP tasks, you can refer to our survey paper which provides an overview of this existing research direction. If you want detailed reference to our library, please refer to our docs.

<!-- [Docs]() | [Graph4nlp survey]() | [Related paper list]() | [Workshops]() -->
- Documentation: [Docs](http://saizhuo.wang/g4nlp/index.html)  
- Graph4NLP Survey: [Graph4nlp survey]()  
- Graph4NLP Tutorials: [Graph4NLP-NAACL'21](https://www.aclweb.org/anthology/2021.naacl-tutorials.3.pdf)([Slides](https://drive.google.com/file/d/1_7cPySt9Pzfd6MaqNihD4FkKI0qzf-s4/view?usp=sharing))  
- Graph4NLP Literature Review: [Literature Lists](https://github.com/graph4ai/graph4nlp_literature )  
- Graph4NLP Workshops : [Workshops](https://deep-learning-graphs.bitbucket.io/dlg-kdd21/index.html)  


## Contributing

Please let us know if you encounter a bug or have any suggestions by filing an issue.

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs. 

## Citation

If you found this code useful, please consider citing the following paper (please stay tuned!).

<!-- Yu Chen, Lingfei Wu and Mohammed J. Zaki. **"Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust
Node Embeddings."** In *Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Dec
6-12, 2020.*

    @article{chen2020iterative,
      title={Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings},
      author={Chen, Yu and Wu, Lingfei and Zaki, Mohammed},
      journal={Advances in Neural Information Processing Systems},
      volume={33},
      year={2020}
    } -->

## Team
Graph4AI Team. Lingfei Wu, Yu Chen, Kai Shen, Xiaojie Guo, Hanning Gao, Shucheng Li, Saizhuo Wang

## License
Graph4NLP uses Apache License 2.0.
