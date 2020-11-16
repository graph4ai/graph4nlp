# Results On Jobs640

## Which min_freq is best for this dataset.
| SAGE+Dynamic_node_emb | bi_fuse | undirected |  
| ---- | ---- | ---- | 
| min_freq=5 | 93.6 | 90.7 |  
| min_freq=4 | 90.7 | 92.9 | 
| min_freq=3 | 90.7 | 92.1 | 
| min_freq=2 | 90.0 | 92.1 | 
| min_freq=1 | 90.0 | 94.3 | 

| SAGE+Constituency | bi_fuse | undirected |  
| ---- | ---- | ---- | 
| min_freq=5 | 90.7 | 90.7 |  
| min_freq=4 | 90.7 | 89.3 | 
| min_freq=3 | 89.3 | 90.7 | 
| min_freq=2 | 90.0 | 88.6| 
| min_freq=1 | 91.4 | 90.0 | 

## All test results in this dataset.
| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | 89.3 | 92.1 | 90.0 |  
| Constituency | 91.4 | 92.1 | 90.0 |
| Dynamic_node_emb | 93.6 | 92.9 | 92.1 |

| GAT | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | 92.9 | 92.9 | 92.9 |  
| Constituency | 92.9 | 93.6 | 92.9 |
| Dynamic_node_emb | 92.1 | 91.4 | 92.1 |

| GGNN | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | 90.7 | 91.4 | 87.9 |  
| Constituency | 89.3 | 93.6 | 87.1 |
| Dynamic_node_emb | 92.9 | 92.1 | 92.1 |

| GCN | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | 89.3 | 90.0 | 89.3 |  
| Constituency | 89.3 | 90.0 | 87.9 |
| Dynamic_node_emb | 90.7 | 90.7 | 90.0 |

## Some exploration for ``dynamic node embedding refine method`` to construct input graph.
| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Gynamic_node_emb_refined | 87.9 | 92.9 | 90.0 |  