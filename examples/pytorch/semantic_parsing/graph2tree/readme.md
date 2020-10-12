# Results On Jobs640

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

<!-- SAGE + undirected + constituency   -->