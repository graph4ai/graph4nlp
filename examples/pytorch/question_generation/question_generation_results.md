Question generation results
============

- Paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)


Dependencies
------------
- torch v1.0: the autograd support for sparse mm is only available in v1.0.
- requests
- sklearn

```bash
pip install torch==1.1.0 requests dgl
```



SQuAD-split2 Results
-------

| GraphType\GNN  |  GAT-Undirected   |  GAT-BiSep    | GAT-BiFuse   | GraphSAGE-Undirected   |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-Undirected   |  GGNN-BiSep    | GGNN-BiFuse   | 
| ------------- |  -------------| ------------- |  -------------|  ------------- | ------------- |  -------------| ------------- | -------------  | ------------- |  
| Dependency     |  |   |  | | |    |  |  |   |
| Constituency (word & non-word nodes) |  |  | |  | |  | ||   |
| NodeEmb | N/A  | N/A | N/A | | - | -  |  | - |  - |
| NodeEmbRefined (line) | N/A  | N/A | N/A |  |- |   -|  | - | -  |
| NodeEmbRefined (dependency) | N/A  | N/A | N/A | |- |   -|  | - | -  |
| NodeEmbRefined (constituency) | N/A  | N/A | N/A |  |- |   -|  | - | -  |






How to run
----------

Run with following:

```java
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```




