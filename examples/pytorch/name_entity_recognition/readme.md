Name entity recognition results
============

How to run
----------



#### Train the model:
```python
python -m examples.pytorch.name_entity_recognition.main.py -json_config examples/pytorch/name_entity_recognition/config/XYZ.json
```

#### Run inference_advance:
```python
python -m examples.pytorch.name_entity_recognition.inference_advance.py -json_config examples/pytorch/name_entity_recognition/config/XYZ.json
```

#### Run inference:
```python
python -m examples.pytorch.name_entity_recognition.inference.py -json_config examples/pytorch/name_entity_recognition/config/XYZ.json
```




Results
-------


| GraphType\GNN  |   GAT-BiSep   |   GAT-BiFuse  |   GAT-Uni      |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |    GraphSAGE-Uni   |  GCN-BiSep    | GCN-BiFuse    | GCN-Uni       | 
| -------------  | ------------- | --------------|----------------| ------------------- | -----------------  | -----------------  |-------------- | ------------- | ------------- |  
| Line_graph     |     79.21     |      78.99    |    79.92       |        80.02        |       80.17        |       79.08        |      77.56    |     78.01     |    77.26      |
| Dependency     |     79.69     |      78.30    |    78.01       |        80.15        |       80.03        |       79.19        |      78.91    |     79.37     |               |
| NodeEmb        |     N/A       |      N/A      |    N/A         |        77.95        |       77.61        |       77.98        |      78.91    |     79.37     |               |
| NodeEmbRefined |     N/A       |      N/A      |    N/A         |        78.15        |       77.93        |       78.02        |               |               |               |




