Name entity recognition results
============


- GraphIE: A Graph-Based Framework for Information Extraction: [https://arxiv.org/abs/1810.13083](https://arxiv.org/abs/1810.13083)



Dependencies
------------
- torch v1.2
- requests
- sklearn
- nltk

```bash
pip install torch==1.2.0 requests dgl
```



Conll Results
-------

| GraphType\GNN  | GAT-Undirected   |  GAT-BiSep    | GAT-BiFuse   | GraphSAGE-Undirected   |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-Undirected   |  GGNN-BiSep    | GGNN-BiFuse   | 
| ------------- |  -------------| ------------- |  -------------|  ------------- | ------------- |  -------------| ------------- | -------------  | ------------- |  
| Dependency_graph     |76.04%|75.53%|75,89%|78.11%|77.53%  |77.55%|77.17%|77.05%|77.10%|
| Line_graph        |75.89% |76.29%|76.27%|77.59%|76.92%|76.97%|76.13%|73.20%|75.75%|
| NodeEmb | N/A  | N/A | N/A | - | - | -  |  | - |  - |
|Only_bilstm| 75.64%|






How to run
----------

Run with following:


Dpendency_graph (graphsage):
```python
python -m examples.pytorch.name_entity_recognition.run_ner --seq_info_encode_strategy bilstm  --graph_type dependency_graph --gpu 0 --init_hidden_size 400 --hidden_size 128 --drop 0.2 --lr 0.01 --batch_size 100 --gnn_type graphsage --direction_option undirected
```
Dpendency_graph (ggnn):
```python
python -m examples.pytorch.name_entity_recognition.run_ner --seq_info_encode_strategy bilstm  --graph_type dependency_graph --gpu 0 --init_hidden_size 400 --hidden_size 128 --drop 0.2 --lr 0.01 --batch_size 100 --gnn_type ggnn --direction_option undirected
```

Dpendency_graph (gat):
```python
python -m examples.pytorch.name_entity_recognition.run_ner --seq_info_encode_strategy bilstm  --graph_type dependency_graph --gpu 0 --init_hidden_size 400 --hidden_size 128 --drop 0.2 --lr 0.001 --batch_size 100 --gnn_type gat --direction_option undirected 
```

Line_graph (graphsage):
```python
python -m examples.pytorch.name_entity_recognition.run_ner --seq_info_encode_strategy bilstm  --graph_type line_graph --gpu 0 --init_hidden_size 400 --hidden_size 128 --drop 0.2 --lr 0.01 --batch_size 100 --gnn_type graphsage --direction_option undirected
```
Line_graph (ggnn):
```python
python -m examples.pytorch.name_entity_recognition.run_ner --seq_info_encode_strategy bilstm  --graph_type line_graph --gpu 0 --init_hidden_size 400 --hidden_size 128 --drop 0.2 --lr 0.01 --batch_size 100 --gnn_type ggnn --direction_option undirected
```

Line_graph (gat):
```python
python -m examples.pytorch.name_entity_recognition.run_ner --seq_info_encode_strategy bilstm  --graph_type line_graph --gpu 0 --init_hidden_size 400 --hidden_size 128 --drop 0.2 --lr 0.001 --batch_size 100 --gnn_type gat --direction_option undirected 
```

bilstm
```python
python -m examples.pytorch.name_entity_recognition.run_ner --seq_info_encode_strategy bilstm  --gpu 0 --init_hidden_size 400 --hidden_size 128 --drop 0.2 --lr 0.01 --batch_size 100
```






