How to run
----------

#### Prepocess:
+ Run the preprocessing script for WN18RR and Kinship: 
```bash
sh examples/pytorch/kg_completion/preprocess.sh
```
+ You can now run the model

#### Run the model:

If you run the task for the first time, remember to set `preprocess: True ` in the config file.

Then run:
```bash
python examples/pytorch/kg_completion/main.py -task_config examples/pytorch/kg_completion/config/kinship.yaml
```

If you want to evaluate the saved model, run:
```bash
python examples/pytorch/kg_completion/inference_advance.py -task_config examples/pytorch/kg_completion/config/kinship.yaml
```

If you want to test the model with a single example:
```bash
python examples/pytorch/kg_completion/inference.py -task_config examples/pytorch/kg_completion/config/kinship.yaml
```
Results on kinship
------------------

Use BCELoss+GGNNDistmult:

| Metrics  |       uni       |      bi_fuse      |   bi_sep    |
| -------- | --------------- | ----------------- | ----------- |
| Hits @1  |    40.4/41.5    |     39.4/41.7     |  38.2/40.3  |
| Hits @10 |    88.3/89.6    |     88.8/89.4     |  88.9/89.1  |
|   MRR    |    54.9/55.9    |     54.8/56.3     |  53.4/55.4  |


Use BCELoss+GCNDistmult:

| Metrics  |      uni        |      bi_fuse      |   bi_sep    |
| -------- | --------------- | ----------------- | ----------- |
| Hits @1  |    39.5/41.9    |     42.9/41.6     |  39.9/41.1  |
| Hits @10 |    88.5/89.6    |     89.2/89.3     |  88.6/89.4  |
|   MRR    |    54.5/56.4    |     56.6/56.4     |  54.6/56.1  |


Use BCELoss+GCNComplex:

| Metrics  |       uni       |      bi_fuse      |   bi_sep    |
| -------- | --------------- | ----------------- | ----------- |
| Hits @1  |    71.2/73.0    |     72.1/73.5     |  73.3/73.9  |
| Hits @10 |    96.8/97.7    |     96.2/97.6     |  97.9/98.0  |
|   MRR    |    80.7/82.3    |     81.6/82.6     |  82.4/82.7  |
