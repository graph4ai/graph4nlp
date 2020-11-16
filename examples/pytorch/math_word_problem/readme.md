# Results on MATHQA

## constituency
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 61.07 | 66.97 | 62.48 | 

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
| GraphSage | 78.4 | 78.0 | 77.2 | 
| GGNN | 76.8 | 77.6 | 76.4 |  
| GAT | 77.2 | 75.2 | 75.6 |  
| GCN | 75.6 | 76.8 | 76.4 |  


## dynamic on node emb refined (constituency as initial graph)
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 76.8 | 77.6 | 77.2 | 
| GGNN | 76.8 | 76.0 | 76.4 |  
| GAT | 76.4 | 76.0 | 75.6 |  
| GCN | 77.2 | 75.6 | 76.4 |  


