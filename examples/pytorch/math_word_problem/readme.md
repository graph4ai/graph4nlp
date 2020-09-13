``python -m examples.pytorch.semantic_parsing.graph2tree.src.runner_mwp -gpuid=3``  

# Results on Mawps

## constituency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 76.4 | 77.2 | 76.8 |  
| GGNN | 77.2 | 76.4 | 74.4 |  
| GAT | 74.8 | 76.0 | 75.6 |  

## dependency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 75.6 | 76.0 | 74.0 | 
