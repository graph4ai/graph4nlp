#!/bin/bash
mkdir data/WN18RR
mkdir data/FB15k-237
mkdir data/kinship
mkdir saved_models
tar -xvf data/WN18RR.tar.gz -C data/WN18RR
tar -xvf data/FB15k-237.tar.gz -C data/FB15k-237
tar -xvf data/kinship.tar.gz -C data/kinship
python wrangle_KG.py WN18RR
python wrangle_KG.py FB15k-237
python wrangle_KG.py FB15k-237-attr
python wrangle_KG.py kinship

# Add new dataset
#mkdir data/DATA_NAME
#python wrangle_KG.py DATA_NAME