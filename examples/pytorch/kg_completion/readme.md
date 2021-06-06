Installation 
===========
+ Download the default English model used by **spaCy**, which is installed in the previous step 
```bash
pip install spacy
python -m spacy download en_core_web_sm
```
+ Run the preprocessing script for WN18RR and Kinship: ```sh preprocess.sh```
+ You can now run the model

How to run
----------


#### Run the model:

If you run the task for the first time, run with:
```bash
python examples/pytorch/kg_completion/kinship_main.py --data kinship --model ggnn_distmult --preprocess
python examples/pytorch/kg_completion/WN18RR_main.py --data WN18RR --model gcn_distmult --lr 0.005 --preprocess
```
Then run:
```bash
python examples/pytorch/kg_completion/kinship_main.py --data kinship --model ggnn_distmult
python examples/pytorch/kg_completion/WN18RR_main.py --data WN18RR --model gcn_distmult --lr 0.005
```

Results on kinship
------------------

Use BCELoss+GGNNDistmult:

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    40.4    |     39.4     |  38.2  |
| Hits @10 |    88.3    |     88.8     |  88.9  |
|   MRR    |    54.9    |     54.8     |  53.4  |

Use BCELoss+GCNDistmult:

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    39.5    |     42.9     |  39.9  |
| Hits @10 |    88.5    |     89.2     |  88.6  |
|   MRR    |    54.5    |     56.6     |  54.6  |


Use BCELoss+GCNComplex:

| Metrics  |    uni     |    bi_fuse   | bi_sep |
| -------- | ---------- | ------------ | ------ |
| Hits @1  |    71.2    |     72.1     |  73.3  |
| Hits @10 |    96.8    |     96.2     |  97.9  |
|   MRR    |    80.7    |     81.6     |  82.4  |
