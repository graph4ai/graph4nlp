``python -m examples.pytorch.semantic_parsing.graph2tree.src.runner_mwp -gpuid=3``  

# Results on Mawps

## constituency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | - | - | - |  
| GGNN | 77.2 | 76.4 | 74.4 |  
| GAT | 74.8 | 76.0 | 75.6 |  

## dependency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 75.6 | 76.0 | 74.0 | 
| GGNN | 75.2 | 76.0 | 75.6 |  
| GAT | 74.8 | 76.0 | 75.6 |  
| GCN | 74.8 | 76.0 | 76.4 |  



## dynamic on node emb
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | - | - | - | 
| GGNN | 77.2 | 76.4 | 74.4 |  
| GAT | 74.8 | 76.0 | 75.6 |  

## dynamic on node emb refined
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | - | - | - | 
| GGNN | 77.2 | 76.4 | 74.4 |  
| GAT | 74.8 | 76.0 | 75.6 |  

