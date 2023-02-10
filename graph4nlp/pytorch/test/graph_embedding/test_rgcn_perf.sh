#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
export test_module=graph4nlp.pytorch.test.graph_embedding.run_rgcn
export python_command="python -m"
export config_root=/student/wangsaizhuo/Codes/graph4nlp/graph4nlp/pytorch/test/graph_embedding/rgcn_scripts

test_routine()
{
    for dataset in {aifb,am,bgs,mutag}
    do
        if [ "$1" == true ] ; then
            command_line="${python_command} ${test_module} -config ${config_root}/run_rgcn_${dataset}.yaml --use_old"
        else
            command_line="${python_command} ${test_module} -config ${config_root}/run_rgcn_${dataset}.yaml"
        fi
        echo "running command ${command_line}"
        ${command_line} &
    done
    wait
}


# Test RGCN-Hetero Implementation on dgl benchmarks
test_routine false

# Test RGCN-Homo Implementation on dgl benchmarks
test_routine true
