#!/bin/bash

cd src
mkdir checkpoint_dir
mkdir checkpoint_dir/valid
mkdir output

echo -----------pretrained embedding generating-----------
python pretrained_embedding.py
echo ------------Begin training---------------------------
python graph2seq.py
echo -----------------------------------------------------
python sample_valid.py 
