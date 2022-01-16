GCN
============

- GCN paper link: [https://arxiv.org/pdf/1609.02907v4.pdf](https://arxiv.org/pdf/1609.02907v4.pdf)

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
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gcn --dataset=cora --gpu=0 --direction-option undirected --early-stop --num-layers 3 --epochs 400 --num-hidden 64
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gcn --dataset=cora --gpu=0 --direction-option bi_sep --early-stop --num-layers 2 --epochs 400 --num-hidden 64
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gcn --dataset=cora --gpu=0 --direction-option bi_fuse --early-stop --num-layers 2 --epochs 400 --num-hidden 64
```

#### Citeseer
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gcn --dataset=citeseer --gpu=0 --direction-option undirected --early-stop --num-layers 3 --epochs 400 --num-hidden 64
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=citeseer --gpu=0 --direction-option bi_sep --early-stop --num-hidden 3703 --num-etypes 1
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=citeseer --gpu=0 --direction-option bi_fuse --early-stop --num-hidden 3703 --num-etypes 1
```

#### Pubmed
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_gcn --dataset=pubmed --gpu=0 --direction-option undirected --early-stop --num-layers 2 --epochs 400 --num-hidden 8 --weight-decay=0.001
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=pubmed --gpu=0 --direction-option bi_sep --weight-decay=0.001 --early-stop --num-layers 2 --epochs 400 --num-hidden 8 
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding_learning.run_ggnn --dataset=pubmed --gpu=0 --direction-option bi_fuse --weight-decay=0.001 --early-stop --num-layers 2 --epochs 400 --num-hidden 8 
```

Results
-------

| Dataset  | GCN-undirected |   GCN-BiSep    |   GCN-BiFuse   |
| -------- | -------------- | -------------- | -------------- |
| Cora     | 0.8066 (0.01)  | 0.8022 (0.007) | 0.7762 (0.01)  |
| Citeseer | 0.6856 (0.008) | 0.6802 (0.008) | 0.6848 (0.009) |

* All the accuracy numbers are averaged after 5 random runs.
* `Cora`, `Citeseer` and `Pubmed` are undirected graph datasets. And `ogbn-arxiv` is a directed graph dataset.


TODO
-------

* Fine-tune hyper-parameters for GGNN-BiSep and GGNN-BiFuse.
* Other datasets: Cora, Citeseer and Pubmed are all undirectional graph datasets, which might not be ideal to test Bidirectional GNN models.

