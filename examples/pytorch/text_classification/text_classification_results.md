Text classification results
============

- GAT paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- GAT-BiSep paper link: [https://arxiv.org/abs/1808.07624](https://arxiv.org/abs/1808.07624)
- GAT-BiFuse paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)

- Convolutional Neural Networks for Sentence Classification: [https://arxiv.org/abs/1408.5882](https://arxiv.org/abs/1408.5882)

- TREC: [https://cogcomp.seas.upenn.edu/Data/QA/QC/](https://cogcomp.seas.upenn.edu/Data/QA/QC/)



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

```java
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

```python
python -m examples.pytorch.test_classification.run_text_classifier
```

Results
-------

| Dataset  |   GCN |  GAT   |  GraphSAGE    | GGNN   |
| -------- | ------------- | ------------- | ------------- | ------------- |  
| Jobs     |  |   |   | |

TODO
-------

- early stopping, loading best model, pretrianed glove vectors, hyper-param tuning, ggnn/graphsage, seq_info_encode_strategy (check node ordering)

