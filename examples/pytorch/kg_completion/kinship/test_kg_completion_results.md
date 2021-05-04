KG Completion Test
============

```bash
python -m examples.pytorch.kg_completion.kinship.main -model_config examples/pytorch/kg_completion/kinship/config/gcn.yaml -task_config examples/pytorch/kg_completion/kinship/config/kinship.yaml
```


Results on kinship
------------------

Use BCELoss:

| Metrics  |  DistMult  |  DistMultGNN  |  SACN  |  TransE  |  TransEGNN |  ComplEx  | ComplExGNN |
| -------- | ---------- | ------------- | ------ | -------- | ---------- | --------- | ---------- |
| Hits @1  |    39.5    |      45.1     |  71.3  |   37.7   |    44.4    |   70.7    |    79.6    |
| Hits @10 |    89.1    |      89.9     |  97.6  |   84.1   |    79.4    |   96.8    |    98.4    |
|   MRR    |    54.2    |      59.0     |  81.3  |   52.9   |    56.1    |   80.6    |    87.0    |

<!-- Use SigmoidLoss:

| Metrics  |  DistMult  |  DistMultGNN  |  ComplEx  |
| -------- | ---------- | ------------- | --------- |
| Hits @1  |    38.3    |     95.8      |   75.5    |
| Hits @10 |    89.3    |     96.5      |   98.4    |
|   MRR    |    53.7    |     96.1      |   84.6    | -->

Use BCELoss+GGNN(2L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    43.1    |     44.1     |  30.8  |
| Hits @10 |    87.6    |     88.5     |  83.4  |
|   MRR    |    56.9    |     58.3     |  45.7  |

<!-- Use SigmoidLoss+GGNN(2L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    88.8    |     66.2     |  44.0  |
| Hits @10 |    93.3    |     92.5     |  88.9  |
|   MRR    |    89.9    |     74.6     |  57.6  | -->

Use BCELoss+GraphSage(2L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    41.7    |     43.7     |  43.7  |
| Hits @10 |    89.7    |     89.2     |  88.5  |
|   MRR    |    56.3    |     57.7     |  57.7  |

<!-- Use SigmoidLoss+GraphSage(2L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    97.1    |     97.2     |  97.0  |
| Hits @10 |    97.2    |     97.3     |  97.2  |
|   MRR    |    97.2    |     97.3     |  97.2  | -->

Use BCELoss+GAT(1L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    36.4    |     31.3     |  31.8  |
| Hits @10 |    85.9    |     84.8     |  85.0  |
|   MRR    |    50.8    |     47.0     |  47.7  |

<!-- Use SigmoidLoss+GAT(1L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    91.1    |     97.0     |  78.1  |
| Hits @10 |    93.8    |     97.2     |  87.6  |
|   MRR    |    91.9    |     97.2     |  80.3  | -->

Use BCELoss+GCN(1L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    34.5    |     31.8     |  19.5  |
| Hits @10 |    84.9    |     85.3     |  76.3  |
|   MRR    |    49.6    |     47.9     |  36.7  |
