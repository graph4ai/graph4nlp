GATED GRAPH SEQUENCE NEURAL NETWORKS (GGNN)
============

- GGNN paper link: [https://arxiv.org/pdf/1511.05493.pdf](https://arxiv.org/pdf/1511.05493.pdf)
- BiSep paper link: [https://arxiv.org/abs/1808.07624](https://arxiv.org/abs/1808.07624)
- BiFuse paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)
- DGL GGNN example: [https://github.com/dmlc/dgl/tree/master/examples/pytorch/ggnn](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ggnn)

Dependencies
------------
- torch v1.3.1
- requests
- sklearn

```bash
pip install torch==1.3.1 requests dgl
```

How to run
----------

Run with following:

#### Cora

```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=cora --gpu=0 --direction-option undirected --early-stop --num-etypes 1
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=cora --gpu=0 --direction-option bi_sep --early-stop --num-etypes 1
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=cora --gpu=0 --direction-option bi_fuse --early-stop --num-etypes 1
```

#### Citeseer
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=citeseer --gpu=0 --direction-option undirected --early-stop --num-hidden 3703 --num-etypes 1
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=citeseer --gpu=0 --direction-option bi_sep --early-stop --num-hidden 3703 --num-etypes 1
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=citeseer --gpu=0 --direction-option bi_fuse --early-stop --num-hidden 3703 --num-etypes 1
```

#### Pubmed
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=pubmed --gpu=0 --direction-option undirected --weight-decay=0.001 --early-stop --num-hidden 500 --num-etypes 1
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=pubmed --gpu=0 --direction-option bi_sep --weight-decay=0.001 --early-stop --num-hidden 500 --num-etypes 1
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=pubmed --gpu=0 --direction-option bi_fuse --weight-decay=0.001 --early-stop --num-hidden 500 --num-etypes 1
```

#### ogbn-arxiv

```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=ogbn-arxiv --gpu=0 --direction-option undirected --early-stop --num-hidden 128
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=ogbn-arxiv --gpu=0 --direction-option bi_sep --early-stop --num-hidden 128
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=ogbn-arxiv --gpu=0 --direction-option bi_fuse --early-stop --num-hidden 128
```
Results
-------

| Dataset  |    GGNN-undirected    |   GGNN-BiSep   |   GGNN-BiFuse  |
| -------- | -------------- | -------------- | -------------- |
| Cora     | 0.7916 (0.01)  | 0.7902 (0.007)  | 0.7936 (0.01)  |
| Citeseer | 0.6296 (0.02)  | 0.6024 (0.01)  | 0.6024 (0.03)  |
| Pubmed   | 0.7684 (0.01)  | 0.7736 (0.01)  | 0.7676 (0.01)  |


* All the accuracy numbers are averaged after 5 random runs.
* `Cora`, `Citeseer` and `Pubmed` are undirected graph datasets. And `ogbn-arxiv` is a directed graph dataset.


TODO
-------

* Fine-tune hyper-parameters for GGNN-BiSep and GGNN-BiFuse.
* Other datasets: Cora, Citeseer and Pubmed are all undirectional graph datasets, which might not be ideal to test Bidirectional GNN models.

