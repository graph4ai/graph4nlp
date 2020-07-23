GraphSAGE Networks (SAmple and aggreGat)
============

- GraphSAGE paper link: [https://arxiv.org/pdf/1706.02216.pdf](https://arxiv.org/pdf/1706.02216.pdf)
- BiSep paper link: [https://arxiv.org/abs/1808.07624](https://arxiv.org/abs/1808.07624)
- BiFuse paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)
- DGL GAT example: [https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage)

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
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=cora --gpu=0 --direction-option uni
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=cora --gpu=0 --direction-option bi_sep
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=cora --gpu=0 --direction-option bi_fuse
```

#### Citeseer
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=citeseer --gpu=0 --early-stop  --direction-option uni
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=citeseer --gpu=0 --early-stop  --direction-option bi_sep
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=citeseer --gpu=0 --early-stop  --direction-option bi_fuse
```

#### Pubmed
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=pubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001 --early-stop  --direction-option uni
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=pubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001 --early-stop  --direction-option bi_sep
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=pubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001 --early-stop  --direction-option bi_fuse
```

#### ogbn-arxiv
<!-- ```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=ogbn-arxiv --gpu=0 --early-stop  --epochs 1000 --num-hidden 128 --direction-option uni 
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=ogbn-arxiv --gpu=0 --early-stop  --epochs 1000 --num-hidden 128 --direction-option bi_sep 
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=ogbn-arxiv --gpu=0 --early-stop  --epochs 1000 --num-hidden 128 --direction-option bi_fuse
``` -->

```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=ogbn-arxiv --gpu=0 --early-stop --direction-option uni
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=ogbn-arxiv --gpu=0 --early-stop --direction-option bi_sep
```
```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_graphsage --dataset=ogbn-arxiv --gpu=0 --early-stop --direction-option bi_fuse
```
Results
-------

| Dataset  | GraphSAGE-Uni |GraphSAGE-BiSep|GraphSAGE-BiFuse|
| -------- | --------------| ------------- | -------------- |
| Cora     | 80.08 (0.01)  |               |                |
| Citeseer |               |               |                |
| Pubmed   |               |               |                |
|ogbn-arxiv|               |               |                |


* All the accuracy numbers are averaged after 5 random runs.
* `Cora`, `Citeseer` and `Pubmed` are undirected graph datasets. And `ogbn-arxiv` is a directed graph dataset.

