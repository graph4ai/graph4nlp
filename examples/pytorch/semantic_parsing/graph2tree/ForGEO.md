# Results On Geo880

## Which min_freq is best for this dataset.
| SAGE+Dynamic_node_emb | bi_fuse | undirected |  
| ---- | ---- | ---- | 
| min_freq=5 | 88.6 | - |  
| min_freq=4 | 89.3 | - | 
| min_freq=3 | 91.1 | - | 
| min_freq=2 | 89.6 | - | 
| min_freq=1 | 88.2 | - | 

| SAGE+Constituency | bi_fuse | undirected |  
| ---- | ---- | ---- | 
| min_freq=5 | - | - |  
| min_freq=4 | - | - | 
| min_freq=3 | - | - | 
| min_freq=2 | - | - | 
| min_freq=1 | - | - | 

## All test results in this dataset.
| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | - |

| GAT | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | - |

| GGNN | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | - |

| GCN | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | - |

## Some exploration for ``dynamic node embedding refine method`` to construct input graph.
| SAGE+Gynamic_node_emb_refined | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | - |
