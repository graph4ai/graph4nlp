#!/bin/bash

# Test old RGCN in DGL
export CUDA_VISIBLE_DEVICES=$1
export wd=/student/wangsaizhuo/Codes/dgl/examples/pytorch/rgcn
cd ${wd}
python entity.py -d aifb --wd 0 --gpu 0 &
python entity.py -d mutag --n-bases 30 --gpu 0 &
python entity.py -d bgs --n-bases 40 --gpu 0 &
python entity.py -d am --n-bases 40 --n-hidden 10 --gpu 0 &
wait

# Test RGCN-hetero in DGL
export wd=/student/wangsaizhuo/Codes/dgl/examples/pytorch/rgcn-hetero
cd ${wd}
python3 entity_classify.py -d aifb --testing --gpu 0 &
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0 &
python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 &
python3 entity_classify.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0 &
wait
