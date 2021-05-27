Installation 
===========
+ Download the default English model used by **spaCy**, which is installed in the previous step ```python -m spacy download en_core_web_sm```
+ Run the preprocessing script for WN18RR, FB15k-237, and Kinship: ```sh preprocess.sh```
+ You can now run the model

KG Completion Test
============

If you run the task for the first time, run with:
```bash
python -m examples.pytorch.kg_completion.kinship.main --data kinship --model ggnn_distmult --preprocess
python -m examples.pytorch.kg_completion.WN18RR.main --data WN18RR --model gcn_distmult --lr 0.005 --preprocess
```
Else:
```bash
python -m examples.pytorch.kg_completion.kinship.main --data kinship --model ggnn_distmult
python -m examples.pytorch.kg_completion.WN18RR.main --data WN18RR --model gcn_distmult --lr 0.005
```


Results on kinship
------------------

Use BCELoss+GGNN(2L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    43.1    |     44.1     |  30.8  |
| Hits @10 |    87.6    |     88.5     |  83.4  |
|   MRR    |    56.9    |     58.3     |  45.7  |

Use BCELoss+GraphSage(2L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    41.7    |     43.7     |  43.7  |
| Hits @10 |    89.7    |     89.2     |  88.5  |
|   MRR    |    56.3    |     57.7     |  57.7  |

Use BCELoss+GAT(1L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    36.4    |     31.3     |  31.8  |
| Hits @10 |    85.9    |     84.8     |  85.0  |
|   MRR    |    50.8    |     47.0     |  47.7  |


Use BCELoss+GCN(1L):

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    34.5    |     31.8     |  19.5  |
| Hits @10 |    84.9    |     85.3     |  76.3  |
|   MRR    |    49.6    |     47.9     |  36.7  |
