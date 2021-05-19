# Results for math word problem

## Results on MATHQA

### constituency-mathqa

| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 50.0 | - | - |

### dynamic on node emb-mathqa

| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 64.08 | 60.23 | 61.37 |

## Results on Mawps

### constituency-mawps

| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 76.4 | - | - |  
| GAT | 75.2 | 76.8 | 76.8 |  
| GCN | 75.6 | 76.4 | 76.0 |  

### dependency-mawps

| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 76.4 | 74.8 | 76.8 |  
| GAT | 75.6 | 76.4 | 75.2 |  
| GCN | 76.0 | 76.0 | 76.8 |  

### dynamic on node emb-mawps

| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 78.4 | 78.0 | 77.2 |  
| GAT | 77.2 | 75.2 | 75.6 |  
| GCN | 75.6 | 76.8 | 76.4 |  

### dynamic on node emb refined (constituency as initial graph)

| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GraphSage | 76.8 | 77.6 | 77.2 |  
| GAT | 76.4 | 76.0 | 75.6 |  
| GCN | 77.2 | 75.6 | 76.4 |  
