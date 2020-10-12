``python -m examples.pytorch.semantic_parsing.graph2tree.src.runner_mwp -gpuid=3``  

> torch==1.5.1+cu101
# Results on Mawps

## constituency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 77.2 | 76.4 | 75.6 |  
| GGNN | 77.2 | 76.0 | 75.2 |  
| GAT | 74.8 | 76.6 | 75.6 |  
| GCN | 74.8 | 76.0 | 76.4 |  


## dependency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 75.6 | 74.4 | 76.4 | 
| GGNN | 75.2 | 76.0 | 75.6 |  
| GAT | 74.8 | 76.0 | 75.6 |  
| GCN | 75.6 | 75.6 | 76.4 |  


## dynamic on node emb
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 76.4 | 76.0 | 76.8 | 
| GGNN | 74.8 | 77.2 | 75.6 |  
| GAT | 77.6 | 74.4 | 73.6 |  
| GCN | 72.4 | 73.2 | 72.0 |  


## dynamic on node emb refined (constituency as initial graph)
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 76.4 | 76.8 | 76.0 | 
| GGNN | 76.8 | 76.0 | 76.0 |  
| GAT | 75.6 | 76.0 | 75.2 |  
| GCN | 73.6 | 74.0 | 74.4 |  


