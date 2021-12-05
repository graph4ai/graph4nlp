Graph Attention Networks (GAT)
============

- GAT paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- GAT-BiSep paper link: [https://arxiv.org/abs/1808.07624](https://arxiv.org/abs/1808.07624)
- GAT-BiFuse paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)
- DGL GAT example: [https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat](https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat)

Dependencies
------------
- torch v1.0: the autograd support for sparse mm is only available in v1.0.
- requests
- sklearn

```bash
pip install torch==1.1.0 requests dgl
```

How to run
----------

Run with following:

#### Cora

```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=cora --gpu=0 --direction-option undirected
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=cora --gpu=0 --direction-option bi_sep
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=cora --gpu=0 --direction-option bi_fuse
```

#### Citeseer
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=citeseer --gpu=0 --early-stop  --direction-option uni
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=citeseer --gpu=0 --early-stop  --direction-option bi_sep
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=citeseer --gpu=0 --early-stop  --direction-option bi_fuse
```

#### Pubmed
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=pubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001 --early-stop  --direction-option uni
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=pubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001 --early-stop  --direction-option bi_sep
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=pubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001 --early-stop  --direction-option bi_fuse
```

#### ogbn-arxiv

```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=ogbn-arxiv --gpu=0 --early-stop --direction-option uni
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=ogbn-arxiv --gpu=0 --early-stop --direction-option bi_sep
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gat --dataset=ogbn-arxiv --gpu=0 --early-stop --direction-option bi_fuse
```

Results
-------

| Dataset  |    GAT-Uni    |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ------------- | ------------- | ------------- |
| Cora     | 84.12 (0.13)  | 83.86 (0.36)  | 81.46 (0.71)  |
| Citeseer | 70.70 (0.55)  | 69.78 (0.60)  | 70.12 (0.71)  |
| Pubmed   | 78.46 (0.43)  | 78.46 (0.43)  | 77.10 (0.49)  |
|ogbn-arxiv| 68.71 (0.69)  | 64.47 (0.29)  | 67.83 (1.30)  |


* All the accuracy numbers are averaged after 5 random runs.
* `Cora`, `Citeseer` and `Pubmed` are undirected graph datasets. And `ogbn-arxiv` is a directed graph dataset.


TODO
-------

* Fine-tune hyper-parameters for GAT-BiSep and GAT-BiFuse.
* Other datasets: Cora, Citeseer and Pubmed are all undirectional graph datasets, which might not be ideal to test Bidirectional GNN models.






