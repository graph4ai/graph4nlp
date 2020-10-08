``python -m examples.pytorch.semantic_parsing.graph2tree.src.runner_mwp -gpuid=3``  

> torch==1.5.1+cu101
# Results on Mawps

## constituency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | - | - | - |  
| GGNN | 77.2 | 76.0 | 75.2 |  
| GAT | 74.8 | 76.6 | 75.6 |  
| GCN | 74.8 | 76.0 | 76.4 |  


## dependency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | - | - | - | 
| GGNN | 75.2 | 76.0 | 75.6 |  
| GAT | 74.8 | 76.0 | 75.6 |  
| GCN | 75.6 | 75.6 | 76.4 |  


## dynamic on node emb
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | - | - | - | 
| GGNN | - | - | - |  
| GAT | - | - | - |  
| GCN | - | - | - |  


## dynamic on node emb refined
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | - | - | - | 
| GGNN | - | - | - |  
| GAT | - | - | - |  
| GCN | - | - | - |  


