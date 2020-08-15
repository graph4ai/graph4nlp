KG Completion Test
============

```bash
python -m examples.pytorch.kg_completion.kinship.main
```


Results on kinship
------------------

Use BCELoss:

| Metrics  |  DistMult  |  DistMultGNN  |  SACN  |  TransE  |  TransEGNN |  ComplEx  | ComplExGNN |
| -------- | ---------- | ------------- | ------ | -------- | ---------- | --------- | ---------- |
| Hits @1  |    39.5    |      45.1     |  71.3  |   37.7   |    44.4    |   70.7    |    79.6    |
| Hits @10 |    89.1    |      89.9     |  97.6  |   84.1   |    79.4    |   96.8    |    98.4    |
|   MRR    |    54.2    |      59.0     |  81.3  |   52.9   |    56.1    |   80.6    |    87.0    |

Use SigmoidLoss:

| Metrics  |  DistMult  |  DistMultGNN  |  ComplEx  |
| -------- | ---------- | ------------- | --------- |
| Hits @1  |    38.3    |     95.8      |   75.5    |
| Hits @10 |    89.3    |     96.5      |   98.4    |
|   MRR    |    53.7    |     96.1      |   84.6    |

Use BCELoss+GGNN(1L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    37.8    |     39.4     |  35.0  |
| Hits @10 |    86.9    |     87.1     |  86.0  |
|   MRR    |    52.5    |     53.9     |  50.2  |

Use SigmoidLoss+GGNN(1L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    88.8    |     46.4     |  50.4  |
| Hits @10 |    93.3    |     90.0     |  89.8  |
|   MRR    |    89.9    |     59.9     |  62.3  |

Use BCELoss+GraphSage(2L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    41.7    |     42.6     |  42.3  |
| Hits @10 |    89.7    |     88.6     |  88.3  |
|   MRR    |    56.3    |     56.5     |  56.3  |

Use SigmoidLoss+GraphSage(2L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    97.1    |     97.2     |  97.0  |
| Hits @10 |    97.2    |     97.3     |  97.2  |
|   MRR    |    97.2    |     97.3     |  97.2  |

Use BCELoss+GAT(1L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    36.4    |     36.8     |  30.0  |
| Hits @10 |    85.9    |     86.4     |  83.6  |
|   MRR    |    50.8    |     51.4     |  45.4  |

Use SigmoidLoss+GAT(1L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    91.1    |     97.0     |  78.1  |
| Hits @10 |    93.8    |     97.2     |  87.6  |
|   MRR    |    91.9    |     97.2     |  80.3  |
