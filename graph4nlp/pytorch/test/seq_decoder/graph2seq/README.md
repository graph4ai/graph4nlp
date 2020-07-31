# Graph2Seq
Graph2Seq aims to automatically learn the mapping between structured input like graphs and sequence output. It consists of a graph encoder and a squence decoder, which can be used for various graph-to-seq tasks in NLP or other AI/ML/DL tasks. In this repo, we evaluate this model in **Math Word Problem (MWP)** auto-solving.

## Simple introduction for MWP
Math Word Problem (MWP) solver aims to automatically generate  equations according to natural language problem descriptions. We evaluated our Graph2Seq model in MWP benchmark dataset [MAWPS](https://www.aclweb.org/anthology/N16-1136). A plain sample is shown below.

| Problem Description | Output Equation |
|-|-|
| 0.5 of the cows are grazing grass . 0.75 of the remaining are sleeping and 9 cows are drinking water from the pond . find the total number of cows . | ( ( 0.5 * x ) + ( 0.75 * ( 0.5 * x ) ) ) + 9.0 = x |


To apply our Graph2Seq model. We need to transform word sequences from problem description to graphs as input of our graph encoder. To do graph construction, we generally employ syntactic parsing to get some additional information like constituency parsing tree and then construct the graph with both new nodes from syntactic parsing and original word sequence nodes.

## GetStarted

- Intall required python packages
  
  > pytorch==1.0.0  
  > numpy==1.15.4  
  > networkx==2.2  
  > tqdm==4.28.1  

- Download [GloVe](http://nlp.stanford.edu/data/glove.6B.zip). For this task, We use the "glove.6B.300d.txt" pretrained embedding.
- Start the Stanford CoreNLP server and then run *data/GraphConstruction/constituency.ipynb* for Graph Generation.

## Data

Our graph data are organized in a dictionary form where meaning of different key-value mappings are as follows:

- *g_ids*: a mapping from the node ID to its ID in the graph
- *g_id_features*: a mapping from the node ID to its text features
- *g_adj*: a mapping from the node ID to its adjacent nodes (represented as thier IDs)


## Train
> sh train.sh

## Test
> sh test.sh

## Environment:

* OS:Ubuntu 16.04.4 LTS  
* Gcc version: 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.10)  
* GPU: TITAN Xp
* CUDA: 8.0 