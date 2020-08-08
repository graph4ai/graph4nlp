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
python -m graph4nlp.pytorch.test.example.jobs.graph2seq.main
```

Results
-------

| Dataset  |    GAT-Uni    |   GAT-BiSep   |  GAT-BiFuse   |
| -------- | ------------- | ------------- | ------------- |
| Jobs     |  33.0 |   |  44.0  |

TODO
-------

- early stopping, loading best model, pretrianed glove vectors, hyper-param tuning, ggnn/graphsage, seq_info_encode_strategy (check node ordering)

