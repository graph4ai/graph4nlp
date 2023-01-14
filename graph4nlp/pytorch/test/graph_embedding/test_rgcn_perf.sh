#!/bin/bash

export test_module=graph4nlp.pytorch.test.graph_embedding.run_rgcn
export python_command="python -m"
export config_root=/student/wangsaizhuo/Codes/graph4nlp/graph4nlp/pytorch/test/graph_embedding/rgcn_scripts

test_routine()
{
    for dataset in {aifb,am,bgs,mutag}
    do
        ${python_command} ${test_module} -config ${config_root}/run_rgcn_${dataset}.yaml &
    done
    wait
}


# Test RGCN-Hetero Implementation on dgl benchmarks
git checkout rgcn-integration
test_routine()

# Test RGCN-Homo Implementation on dgl benchmarks
git checkout debug-orig-rgcn
test_routine()
