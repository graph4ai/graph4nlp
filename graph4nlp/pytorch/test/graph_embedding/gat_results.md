Graph Attention Networks (GAT)
============

- GAT paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- GAT-BiSep Paper link: [https://arxiv.org/abs/1808.07624](https://arxiv.org/abs/1808.07624)
- GAT-BiFuse Paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)
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

```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_gat --dataset=cora --gpu=0 --direction-option uni
```

```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_gat --dataset=citeseer --gpu=0 --early-stop  --direction-option uni
```

```bash
python -m graph4nlp.pytorch.test.graph_embedding.run_gat --dataset=pubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001 --early-stop  --direction-option uni
```



Results
-------

| Dataset  |    GAT-Uni    |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ------------- | ------------- | ------------- |
| Cora     | 84.12 (0.13)  | 83.86 (0.36)  | 81.46 (0.71)  |
| Citeseer | 70.26 (0.40)  | 67.84 (1.40)  | 69.54 (0.58)  |
| Pubmed   | 78.46 (0.43)  | 78.42 (0.33)  | 77.76 (0.24)  |


* All the accuracy numbers are averaged after 5 random runs.


