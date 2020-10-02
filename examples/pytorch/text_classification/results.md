Text Classification Test
============

```bash
python -m examples.pytorch.kg_completion.kinship.main
```


Results on CAirline
------------------

| GraphType\GNN  |  GAT-Undirected   |  GAT-BiSep    | GAT-BiFuse   | GraphSAGE-Undirected   |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-Undirected   |  GGNN-BiSep    | GGNN-BiFuse   | 
| -------------- |  -------------| ------------- |  -------------|  ------------- | ------------- |  -------------| ------------- | -------------  | ------------- |  
| Dependency     | 0.7654  | 0.7890  | 0.7843 | 0.946* | 0.944* |  0.942  | 0.7606 | 0.7732 |  0.7811 |
| Constituency   | 0.7795  | 0.7827 | 0.7874 | 0.934 |0.928 | 0.944*  | 0.920 |0.944* |  0.940* |
| NodeEmb | N/A  | N/A | N/A | 0.936 | 0.932 | 0.928  |  | | |
| NodeEmbRefined (dependency) | N/A  | N/A | N/A |0.928 |0.928 | 0.930  |  |  |   |

