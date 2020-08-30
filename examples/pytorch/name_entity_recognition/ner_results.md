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
| Dependency_graph     |   |   |  |  |  |    |  | |   |
| Line_graph        |   |   |  |74.72%|74.40%|73.10%|76.13%| |75.75%|
| NodeEmb | N/A  | N/A | N/A | - | - | -  |  | - |  - |
|Only_bilstm| 75.64%|






How to run
----------

Run with following:


Dependency graph:

bilstm
```python
python -m examples.pytorch.name_entity_recognition.run_ner --seq_info_encode_strategy bilstm  --graph_type line_graph --gpu 0 --init_hidden_size 300 --lstm_hidden_size 80 --drop 0.2 --lr 0.01 --batch_size 150
```






