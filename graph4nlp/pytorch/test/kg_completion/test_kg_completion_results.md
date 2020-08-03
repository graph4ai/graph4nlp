KG Completion Test
============

- The test code is referred to **End-to-end Structure-Aware Convolutional Networks
for Knowledge Base Completion** [pdf](https://arxiv.org/pdf/1811.04441.pdf) [code](https://github.com/JD-AI-Research-Silicon-Valley/SACN)

Dependencies
------------
- PyTorch v1.0
- Install the requirements: ```pip install -r requirements.txt```
- Download the default English model used by [spaCy](https://github.com/explosion/spaCy), which is installed in the previous step ```python -m spacy download en```.

preprocess
----------

Run the preprocessing script for kinship: `sh preprocess.sh`.

How to run
----------

#### DistMult

Default loss_name: BCELoss

```bash
python main.py model DistMult dataset kinship process True loss_name BCELoss
python main.py model DistMult dataset kinship process True loss_name MSELoss
python main.py model DistMult dataset kinship process True loss_name SigmoidLoss
python main.py model DistMult dataset kinship process True loss_name SoftPlusLoss
python main.py model DistMult dataset kinship process True loss_name SoftMarginLoss
```

#### DistMultGNN

```bash
python main.py model DisMultGNN dataset kinship process True loss_name BCELoss
```

#### TransE

```bash
python main.py model TransE dataset kinship process True loss_name BCELoss
```

#### TransEGNN

```bash
python main.py model TransEGNN dataset kinship process True loss_name BCELoss
```

#### ComplEx

```bash
python main.py model ComplEx dataset kinship process True loss_name BCELoss
```

#### ComplExGNN

```bash
python main.py model ComplExGNN dataset kinship process True loss_name BCELoss
```

#### SACN

```bash
python main.py model SACN dataset kinship process True loss_name BCELoss
```


Results on kinship
------------------
Reference results from [Multi-Hop Knowledge Graph Reasoning with Reward Shaping](https://www.aclweb.org/anthology/D18-1362.pdf)

| Metrics  |  DistMult  |  ComplEx  |  ConvE  |
| -------- | ---------- | --------- | ------- |
| Hits @1  |    48.7    |   81.8    |   79.7  |
| Hits @10 |    90.4    |   98.1    |   98.1  |
|   MRR    |    61.4    |   88.4    |   87.1  |

Use BCELoss:

| Metrics  |  DistMult  |  DistMultGNN  |  SACN  |  TransE  |  TransEGNN |  ComplEx  | ComplExGNN |
| -------- | ---------- | ------------- | ------ | -------- | ---------- | --------- | ---------- |
| Hits @1  |    39.5    |      45.1     |  71.3  |   37.7   |    44.4    |   70.7    |    79.6    |
| Hits @10 |    89.1    |      89.9     |  97.6  |   84.1   |    79.4    |   96.8    |    98.4    |
|   MRR    |    54.2    |      59.0     |  81.3  |   52.9   |    56.1    |   80.6    |    87.0    |

Use MSELoss:

| Metrics  |  DistMult  |  DistMultGNN |  TransEGNN  |
| -------- | ---------- | ------------ | ----------- |
| Hits @1  |    39.8    |     44.4     |    27.2     |
| Hits @10 |    88.3    |     87.4     |    57.0     |
|   MRR    |    54.1    |     58.0     |    37.1     |

Use SoftplusLoss:

| Metrics  |  DistMult  |  DistMultGNN  |  ComplEx  | ComplExGNN |
| -------- | ---------- | ------------- | --------- | ---------- |
| Hits @1  |    39.1    |     96.7      |   76.6    |    96.9    |
| Hits @10 |    88.3    |     96.7      |   97.9    |    97.1    |
|   MRR    |    53.9    |     96.9      |   85.0    |    97.0    |

Use SigmoidLoss:

| Metrics  |  DistMult  |  DistMultGNN  |  ComplEx  |
| -------- | ---------- | ------------- | --------- |
| Hits @1  |    38.3    |     95.8      |   75.5    |
| Hits @10 |    89.3    |     96.5      |   98.4    |
|   MRR    |    53.7    |     96.1      |   84.6    |

Use SoftMarginLoss:

| Metrics  |  DistMult  |  DistMultGNN  |
| -------- | ---------- | ------------- |
| Hits @1  |    39.6    |    overfit    |
| Hits @10 |    88.9    |    overfit    |
|   MRR    |    54.5    |    overfit    |