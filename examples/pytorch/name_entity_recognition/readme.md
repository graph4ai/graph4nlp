Name entity recognition results
============


Conll Results
-------

| GraphType\GNN  | GAT-Undirected   |  GAT-BiSep    | GAT-BiFuse   | GraphSAGE-Undirected   |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-Undirected   |  GGNN-BiSep    | GGNN-BiFuse   | GCN-Undirected   |  GCN-BiSep    | GCN-BiFuse   |
| ------------- |  -------------| ------------- |  -------------|  ------------- | ------------- |  -------------| ------------- | -------------  | ------------- | ------------- | -------------  | ------------- |  
| Dependency_graph     |76.04%|75.53%|75,89%|78.11%|77.53%  |77.55%|77.17%|77.05%|77.10%|77.66%|74.82%|76.03%|
| Line_graph        |75.89% |76.29%|76.27%|81.06%|79.35%|80.50%|76.13%|73.20%|75.75%|75.71%|75.63%|76.51%|
| NodeEmb | N/A  | N/A | N/A | 78.07% | 77.69% | 77.55%  |73.86%| 72.59%| 75.58% |75.33%|74.74%|74.87%|
| NodeEmb_line | N/A  | N/A | N/A | 76.44% |77.27%|76.23%  |74.39%|75.46%|75.23%  |75.29%|76.03%|75.13%|
| NodeEmb_dependency | N/A  | N/A | N/A |76.45%|78.21%  |79.58%|73.30%|72.39%|73.40%  |75.87%|74.02%|73.50%|
|Only_bilstm| 75.64%||||






How to run
----------

#### Start the StanfordCoreNLP server for data preprocessing:

1) Download StanfordCoreNLP `https://stanfordnlp.github.io/CoreNLP/`
2) Go to the root folder and start the server:

```java
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```


#### Run with following:


Dpendency_graph (graphsage):
```python
python examples/pytorch/name_entity_recognition/main.py   --graph_type dependency_graph --gpu 0 --init_hidden_size 400 --hidden_size 128  --lr 0.01 --batch_size 100 --gnn_type graphsage --direction_option undirected
```
Dpendency_graph (ggnn):
```python
python examples/pytorch/name_entity_recognition/main.py   --graph_type dependency_graph --gpu 0 --init_hidden_size 400 --hidden_size 128  --lr 0.01 --batch_size 100 --gnn_type ggnn --direction_option undirected
```
Dpendency_graph (gcn):
```python
python examples/pytorch/name_entity_recognition/main.py   --graph_type dependency_graph --gpu 0 --init_hidden_size 400 --hidden_size 128  --lr 0.001 --batch_size 100 --gnn_type gcn --direction_option undirected
```

Dpendency_graph (gat):
```python
python examples/pytorch/name_entity_recognition/main.py   --graph_type dependency_graph --gpu 0 --init_hidden_size 400 --hidden_size 128 2 --lr 0.001 --batch_size 100 --gnn_type gat --direction_option undirected 
```

Line_graph (graphsage):
```python
python examples/pytorch/name_entity_recognition/main.py   --graph_type line_graph --gpu 0 --init_hidden_size 400 --hidden_size 128  --lr 0.01 --batch_size 100 --gnn_type graphsage --direction_option undirected
```
Line_graph (gcn):
```python
python examples/pytorch/name_entity_recognition/main.py   --graph_type line_graph --gpu 0 --init_hidden_size 400 --hidden_size 128  --lr 0.001 --batch_size 100 --gnn_type gcn --direction_option undirected
```

Line_graph (ggnn):
```python
python examples/pytorch/name_entity_recognition/main.py  --graph_type line_graph --gpu 0 --init_hidden_size 400 --hidden_size 128  --lr 0.01 --batch_size 100 --gnn_type ggnn --direction_option undirected
```

Line_graph (gat):
```python
python examples/pytorch/name_entity_recognition/main.py  --graph_type line_graph --gpu 0 --init_hidden_size 400 --hidden_size 128 --lr 0.001 --batch_size 100 --gnn_type gat --direction_option undirected 
```

Dynamic_graph (graphsage):
```python
python examples/pytorch/name_entity_recognition/main.py  --graph_type node_emb --gpu 0 --init_hidden_size 400 --hidden_size 128  --lr 0.01 --batch_size 100 --gnn_type graphsage --direction_option undirected
```

Dynamic_graph (gcn):
```python
python examples/pytorch/name_entity_recognition/main.py   --graph_type node_emb --gpu 0 --init_hidden_size 400 --hidden_size 128  --lr 0.001 --batch_size 100 --gnn_type gcn --direction_option undirected
```

Dynamic_graph (ggnn):
```python
python examples/pytorch/name_entity_recognition/main.py   --graph_type node_emb --gpu 0 --init_hidden_size 400 --hidden_size 128  --lr 0.01 --batch_size 100 --gnn_type ggnn --direction_option undirected
```

Dynamic_graph (gat):
```python
python examples/pytorch/name_entity_recognition/main.py   --graph_type node_emb --gpu 0 --init_hidden_size 400 --hidden_size 128 2 --lr 0.001 --batch_size 100 --gnn_type gat --direction_option undirected 
```

bilstm
```python
python examples/pytorch/name_entity_recognition/main.py   --gpu 0 --init_hidden_size 400 --hidden_size 128  --lr 0.01 --batch_size 100 --use_gnn False
```


