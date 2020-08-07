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

#### ogbg-molhiv

```bash
python -m graph4nlp.pytorch.test.graph_classification.run_graph_classification --dataset=ogbg-molhiv --graph-pooling avg_pool --device 0
```

Results
-------

| Dataset  |    AvgPooling    | MaxPooling | 
| -------- | ------------- | ------------- | 
| MiniGC | 12.5 | 28.8 |

<!-- 
Avg_pool
Accuracy of sampled predictions on the test set: 15.0000%
Accuracy of argmax predictions on the test set: 12.500000%    

Max_pool (no linear projection)
Accuracy of sampled predictions on the test set: 16.2500%
Accuracy of argmax predictions on the test set: 28.750000% 
 -->


<!-- TODO
-------

 -->





