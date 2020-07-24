# SACN

Paper: "[End-to-end Structure-Aware Convolutional Networks for Knowledge Base Completion](https://arxiv.org/pdf/1811.04441.pdf)" 

Published in the Thirty-Third AAAI Conference on Artificial Intelligence ([AAAI-19](https://aaai.org/Conferences/AAAI-19/)). 

--- PyTorch Version ---

## Overview
The end-to-end Structure-Aware Convolutional Network (SACN) model takes the benefit of GCN and ConvE together for knowledge base completion. SACN consists of an encoder of a weighted graph convolutional network (WGCN), and a decoder of a convolutional network called Conv-TransE. WGCN utilizes knowledge graph node structure, node attributes and
edge relation types. The decoder Conv-TransE enables the state-of-the-art ConvE to be translational between entities and relations while keeps the same link prediction performance as ConvE. 

## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch 1.0 ](https://github.com/pytorch/pytorch) using [official website](https://pytorch.org/) or [Anaconda](https://www.continuum.io/downloads). 

2. Install the requirements: `pip install -r requirements.txt`

3. Download the default English model used by [spaCy](https://github.com/explosion/spaCy), which is installed in the previous step `python -m spacy download en`.

## Data Preprocessing

Run the preprocessing script for FB15k-237, WN18RR, FB15k-237-attr and kinship: `sh preprocess.sh`.

## Run a model

To run a model, you first need to preprocess the data. This can be done by specifying the `process` parameter.  

For ConvTransE model, you can run it using:
```
CUDA_VISIBLE_DEVICES=0 python main.py model ConvTransE init_emb_size 100 dropout_rate 0.4 channels 50 lr 0.001 kernel_size 3 dataset FB15k-237 process True
```
For SACN model, you can run it using:
```
CUDA_VISIBLE_DEVICES=0 python main.py model SACN dataset FB15k-237 process True
```

You can modify the hyper-parameters from "src.spodernet.spodernet.utils.global_config.py" or specify the hyper-parameters in the command. For different datasets, you need to tune the parameters. 

For this test version, if you find any problems, please feel free and email me. We will keep updating the code.

## Acknowledgements

Code is inspired by [ConvE](https://github.com/TimDettmers/ConvE). 

