Text classification results
============

How to run
----------



#### Train the model:
```python
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/trec/XYZ.yaml
```

#### Run inference:
```python
python -m examples.pytorch.text_classification.inference_advance -config examples/pytorch/text_classification/config/trec/XYZ.yaml
```






TREC Results
-------


| GraphType\GNN  |   GAT-BiSep   |   GAT-BiFuse  |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-BiSep   | GGNN-BiFuse   | 
| -------------  | ------------- | --------------| ------------------- | -----------------  |-------------- | ------------- |  
| Dependency     |     0.9480    |   0.9460      |         0.942       |      0.958         |      0.954    |     0.9440    |
| Constituency   |     0.9420    |   0.9300      |         0.952       |      0.950         |      0.952    |     0.9400    |
| NodeEmb        |      N/A      |    N/A        |         0.930       |      0.908         |               |               |
| NodeEmbRefined |      N/A      |    N/A        |         0.940       |      0.926         |               |               |



CAirline Results
-------


| GraphType\GNN  |  GAT-BiSep   |  GGNN-BiSep   |GraphSage-BiSep| 
| -------------- | ------------ | ------------- |---------------|
| Dependency     | 0.7496       | 0.8020        | 0.7977        |
| Constituency   | 0.7846       | 0.7933        | 0.7948        |
| NodeEmb        | N/A          | 0.8108        | 0.8108        |
| NodeEmbRefined | N/A          | 0.7991        | 0.8020        |

