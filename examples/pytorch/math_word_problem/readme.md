# Results on MATHQA

## dynamic on node emb
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 64.62 | 64.02 | 61.37 | 

# Results on Mawps

## constituency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 78.0 | 77.2 | 76.4 |  
| GGNN | 77.6 | 76.4 | 74.8 |  
| GAT | 75.2 | 76.8 | 76.8 |  
| GCN | 75.6 | 76.4 | 76.0 |  


## dependency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 76.4 | 74.8 | 76.8 | 
| GGNN | 75.2 | 76.0 | 76.0 |  
| GAT | 75.6 | 76.4 | 75.2 |  
| GCN | 76.0 | 76.0 | 76.8 |  


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


