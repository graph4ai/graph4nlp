Graph2Seq results
============

- GAT paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- GAT-BiSep paper link: [https://arxiv.org/abs/1808.07624](https://arxiv.org/abs/1808.07624)
- GAT-BiFuse paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)



How to run
----------

#### Start the StanfordCoreNLP server for data preprocessing:

1) Download StanfordCoreNLP `https://stanfordnlp.github.io/CoreNLP/`
2) Go to the root folder and start the server:

```java
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

#### Run with following:

```python
python examples/pytorch/semantic_parsing/graph2seq/main.py --dataset_yaml examples/pytorch/semantic_parsing/graph2seq/config/new_dependency_gcn_undirected.yaml
```

#### Inference with following:
```python
python examples/pytorch/semantic_parsing/graph2seq/inference.py --dataset_yaml examples/pytorch/semantic_parsing/graph2seq/config_inference/new_dependency_gcn_undirected.yaml
```

Results
-------

As_node(Dep + GAT): 90.0 90.0 89.1


Dependency

| Dataset  |      GAT-Uni     |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |    95.7(95.7)    |   94.3(94.3)  |      94.3(94.3)     |

| Dataset  |      GGNN-Uni    |   GGNN-BiSep  |  GGNN-BiFuse  |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       94.3(94.3)       |   93.6(93.6)    |      94.3(95.0)    |

| Dataset  |  Graphsage-Uni   | Graphsage-BiSep   |  Graphsage-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       94.3(94.3)     |   93.6(93.6)  |  94.3(94.3)   |
| w/o. copy | 90.0(90.0) | 92.9(92.9) | 92.9(92.1) |
| w/o. coverage | 90.7(90.7) | 87.1(86.4) | 90.7(90.7) |
| w/o. copy&coverage| 89.3(89.3) | 89.3(89.3) | 88.6(87.1) |

| Dataset  |      GCN-Uni     |   GCN-BiSep   |  GCN-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |     93.6(93.6)   |    92.9(92.9) |   93.6(93.6)  |


Cos

| Dataset  |      GAT-Uni     |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |     95.0(95.0)   |     89.3(89.3)      |     92.1(92.1)      |

| Dataset  |      GGNN-Uni    |   GGNN-BiSep  |  GGNN-BiFuse  |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       94.3(94.3)       |      95.7(95.7)     |     92.9(92.9)      |

| Dataset  |  Graphsage-Uni   | Graphsage-BiSep   |  Graphsage-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       93.6(93.6)       |     95.0(95.0)      |      94.3(94.3)     |

| Dataset  |      GCN-Uni     |   GCN-BiSep   |  GCN-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |     92.9(92.9)   |    93.6(93.6) |   89.3(89.3)  |


Dynamic: node_emb

| Dataset  |      GAT-Uni     |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       92.9(92.9)       |    93.6(93.6)       |     94.3(94.3)      |

| Dataset  |      GGNN-Uni    |   GGNN-BiSep  |  GGNN-BiFuse  |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       92.1(92.1)      |      92.9(92.9)     |     93.6(93.6)      |

| Dataset  |  Graphsage-Uni   | Graphsage-BiSep   |  Graphsage-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       94.3(94.3)       |     92.1(92.1)      |      92.1(92.1)     |

| Dataset  |  GCN-Uni   | GCN-BiSep   |  GCN-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       91.4(90.7)       |     90.7(90.7)      |      91.4(91.4)     |

Dynamic Refine(dep initial 0.2)

| Dataset  |      GAT-Uni     |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       92.1(92.1)       |    92.1(92.1)       |     92.1(91.4)      |

| Dataset  |      GGNN-Uni    |   GGNN-BiSep  |  GGNN-BiFuse  |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       94.3(94.3)      |       93.6(93.6)     |      93.6(93.6)      |

| Dataset  |  Graphsage-Uni   | Graphsage-BiSep   |  Graphsage-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       94.3(94.3)       |     94.3(94.3)     |      92.9(92.9)     |

| Dataset  |  GCN-Uni   | GCN-BiSep   |  GCN-BiFuse   |
| -------- | ---------------- | ------------- | ------------- |
| Jobs     |       93.6(93.6)       |     93.6(93.6)      |      93.6(93.6)     |



