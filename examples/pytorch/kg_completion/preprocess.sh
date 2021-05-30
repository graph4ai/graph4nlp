#!/bin/bash
mkdir examples/pytorch/kg_completion/data
mkdir examples/pytorch/kg_completion/data/WN18RR
mkdir examples/pytorch/kg_completion/data/kinship
mkdir examples/pytorch/kg_completion/saved_models
tar -xvf examples/pytorch/kg_completion/WN18RR/raw/WN18RR.tar.gz -C examples/pytorch/kg_completion/data/WN18RR
tar -xvf examples/pytorch/kg_completion/kinship/raw/kinship.tar.gz -C examples/pytorch/kg_completion/data/kinship
python examples/pytorch/kg_completion/wrangle_KG.py WN18RR
python examples/pytorch/kg_completion/wrangle_KG.py kinship
