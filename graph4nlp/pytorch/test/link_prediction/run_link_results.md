Test for link prediction module
============

- This test code is based on the task of GVAE: [https://arxiv.org/abs/1611.07308](https://arxiv.org/abs/1611.07308)
Three kinds of link prediction methods are tested: ElementSum, StackedElementProd, ConcatFeedForwardNN.
Dependencies
------------
- torch v1.0: the autograd support for sparse mm is only available in v1.0.
- scipy
- sklearn
- dgl

```bash
pip install torch==1.1.0 requests dgl
```

How to run
----------

Run with following:

#### Cora

```bash
python -m graph4nlp.pytorch.test.link_prediction.link_prediction_test --dataset-str=cora --prediction_type elesum
```
```bash
python -m graph4nlp.pytorch.test.link_prediction.link_prediction_test --dataset-str=cora --prediction_type concat_NN
```
```bash
python -m graph4nlp.pytorch.test.link_prediction.link_prediction_test --dataset-str=cora --prediction_type stacked_ele_prod
```

#### Citeseer
```bash
python -m graph4nlp.pytorch.test.link_prediction.link_prediction_test --dataset-str=citeseer --prediction_type elesum
```
```bash
python -m graph4nlp.pytorch.test.link_prediction.link_prediction_test --dataset-str=citeseer --prediction_type concat_NN
```
```bash
python -m graph4nlp.pytorch.test.link_prediction.link_prediction_test --dataset-str=citeseer --prediction_type stacked_ele_prod
```

Results
-------

| Dataset  |  ElementSum  |   ConcatFFNN  | StackedElementProd | Baseline (InnerProd) |
| -------- | -------------| ------------- | ------------------ |----------------------|
| Cora     | 88.70 (89.94)| 89.03 (90.87) |    90.38 (91.89)   |    91.13 (91.53)     |
| Citeseer | 86.10 (87.58)| 85.39 (87.42) |    92.21 (93.81)   |    90.80 (92.00)     |


* The metrics are: AUC (precision)*100 for the binary classification task.
