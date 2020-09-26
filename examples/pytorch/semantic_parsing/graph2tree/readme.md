# Results On Jobs640

## Constituency Graph
| Encoder\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |
| GAT | 89.3 | 92.1 | 90.7 |  
| GGNN | 91.4 | 91.4 | 90.0 |
| GraphSage | 90.7 | 90.7 | 90.0 |

## Dependency Graph
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GAT | 90.7 | 91.4 | 90.0 |  
| GGNN | 89.3 | 90.0 | 90.7 |  
| GraphSage | 90.0 | 90.7 | 89.3 |  

## Dynamic Graph
| Methods\fuse_strategy | undirected | bi_sep | bi_fuse |  
| ---- | ---- | ---- | ---- |  
| GAT | 90.0 | 90.7 | 91.4 |  
| GGNN | 89.3 | 87.9 | 87.9 |  
| GraphSage | 87.9 | 90.0 | 91.4 |  

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