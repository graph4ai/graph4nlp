KG Completion Test
==================

preprocess
----------

Run the preprocessing script for kinship: `sh preprocess.sh`.

How to run
----------

#### DistMult

```bash
python main.py model DistMult dataset kinship process True
```

#### DistMultGNN

```bash
python main.py model DisMultGNN dataset kinship process True
```

#### SACN

```bash
python main.py model SACN dataset kinship process True
```

#### TransE

```bash
python main.py model TransE dataset kinship process True
```

#### TransEGNN

```bash
python main.py model TransEGNN dataset kinship process True
```

Results on kinship
------------------

| Metrics  |  DistMult  |  DistMultGNN  |  SACN  |  TransE  |  TransEGNN |
| -------- | ---------- | ------------- | ------ | -------- | ---------- |
| Hits @1  |    0.40    | 0.458         | 0.713  | 0.354    | 0.427      |
| Hits @10 |    0.87    | 0.905         | 0.976  | 0.827    | 0.796      |
|   MRR    |    0.55    | 0.598         | 0.813  | 0.5078   | 0.551      |

