Graph2Seq results
============

- GAT paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- GAT-BiSep paper link: [https://arxiv.org/abs/1808.07624](https://arxiv.org/abs/1808.07624)
- GAT-BiFuse paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)


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

```python
python -m examples.pytorch.semantic_parsing.graph2seq.main --dataset_yaml examples/pytorch/semantic_parsing/graph2seq/config/dependency.yaml
```

Results
-------

As_node(Dep + GAT): 90.0 90.0 89.1


Dependency

| Dataset  |      GAT-Uni     |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       91.4       |       92.9    |      90.0     |

| Dataset  |      GGNN-Uni    |   GGNN-BiSep  |  GGNN-BiFuse  |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       92.9       |       91.4    |      91.4     |

| Dataset  |  Graphsage-Uni   | Graphsage-BiSep   |  Graphsage-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       92.1       |       92.1    |      90.7     |

| Dataset  |      GCN-Uni     |   GCN-BiSep   |  GCN-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       91.4       |       91.4    |      92.1     |


Cos

| Dataset  |      GAT-Uni     |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       91.4       |     91.4      |     92.1      |

| Dataset  |      GGNN-Uni    |   GGNN-BiSep  |  GGNN-BiFuse  |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       91.4       |      91.4     |     90.0      |

| Dataset  |  Graphsage-Uni   | Graphsage-BiSep   |  Graphsage-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       90.7       |     90.7      |      90.7     |

| Dataset  |      GCN-Uni     |   GCN-BiSep   |  GCN-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       90.7       |      91.4    |      90.0     |


Dynamic: node_emb

| Dataset  |      GAT-Uni     |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       92.1       |    92.9       |     90.0      |

| Dataset  |      GGNN-Uni    |   GGNN-BiSep  |  GGNN-BiFuse  |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       92.1      |      93.6     |     93.6      |

| Dataset  |  Graphsage-Uni   | Graphsage-BiSep   |  Graphsage-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       92.9       |     92.1      |      92.9     |

Dynamic Refine(dep initial 0.2)

| Dataset  |      GAT-Uni     |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       91.4       |    92.9       |     90.0      |

| Dataset  |      GGNN-Uni    |   GGNN-BiSep  |  GGNN-BiFuse  |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       92.9      |       93.1     |      92.9      |

| Dataset  |  Graphsage-Uni   | Graphsage-BiSep   |  Graphsage-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       91.4       |     92.1      |      91.4     |



TODO
-------

- early stopping, loading best model, pretrianed glove vectors, hyper-param tuning, ggnn/graphsage, seq_info_encode_strategy (check node ordering)

- 2e-3

