# Results On Geo

min_freq=6

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | 87.1 |

min_freq=5

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | 87.5 |

min_freq=4

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | 89.3 |

min_freq=3

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | 90.0 |

min_freq=2

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | 88.2 |

min_freq=1

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | 90.7 | - | 88.2 |

# Results On Jobs640

## Results for copy  

min_freq=6

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | - |

min_freq=5

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | - |

min_freq=4

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | - |

min_freq=3

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | 92.9 |

min_freq=2

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | 92.1 |

min_freq=1

| SAGE | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | - | - | - |  
| Constituency | - | - | - |
| Dynamic_node_emb | - | - | 94.3 |


## Results for toBatch fix  
| Encoder\fuse_strategy | DependencyGraph | ConstituencyGraph | DynamicGraph_node_emb | DynamicGraph_node_emb_refined |  
| ---- | ---- | ---- | ---- | ---- |  
| GCN+bi_fuse | 90.0 | 91.4 | 89.3 | 89.3 |

## Test results for GGNN (dgl==0.5.2)

| Graph Construction\fuse_strategy | bi_fuse | bi_sep | undirected |  
| ---- | ---- | ---- | ---- |
| Dependency | 89.3 | 90.7 | 89.3 |  
| Constituency | 90.0 | 90.7 | 89.3 |
| Dynamic_node_emb | 90.0 | 90.7 | 91.4 |
| Dynamic_node_emb_refined | 91.4 | 91.4 | 91.4 |


## Constituency Graph
| Encoder\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |
| GAT | 89.3 | 92.1 | 90.7 |  
| GGNN | 91.4 | 91.4 | 90.0 |
| GraphSage | 90.7 | 90.7 | 90.0 |
| GCN | 89.3 | 91.4 | 90.7 |


## Dependency Graph
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GAT | 90.7 | 91.4 | 90.0 |  
| GGNN | 89.3 | 90.0 | 90.7 |  
| GraphSage | 90.0 | 90.7 | 89.3 |  
| GCN | 90.0 | 90.0 | 91.4 |

## Dynamic Graph node emb
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GAT | 90.0 | 90.7 | 91.4 |  
| GGNN | 89.3 | 87.9 | 87.9 |  
| GraphSage | 87.9 | 90.0 | 91.4 |  
| GCN | 89.3 | 90.0 | 89.3 |


## Dynamic Graph node emb refined (constituency tree as initial graph)
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GAT | 90.0 | 90.7 | 91.4 |  
| GGNN | 89.3 | 87.9 | 87.9 |  
| GraphSage | 91.4 | 90.0 | 91.4 |  
| GCN | 90.7 | 90.0 | 89.3 |

## Ablation study for tree decoder

## Constituency Graph
| Encoder\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |
| GGNN w/ sibling feeding | 91.4 | 91.4 | 90.0 |  
| GGNN w/o sibling feeding | 90.7 | 91.4 | 89.3 |  
| GraphSage w/ sibling feeding | 90.7 | 90.7 | 90.0 |
| GraphSage w/o sibling feeding | 90.7 | 89.3 | 90.7 |

## Use as-node in dependency graph construction

| Methods\fuse_strategy | graph constrcution strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- | ---- |
| GraphSage | only connection | 90.0 | 90.7 | 89.3 |  
| GraphSage | as node | 90.0 | 90.0 | 89.3 |  

## For beam search: constituency graph

| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |
| GraphSage-greedy | 90.0 | 90.7 | 89.3 |  
| GraphSage-beam | 90.7 | 90.7 | 90.0 |  

# Results On Jobs640-new

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



