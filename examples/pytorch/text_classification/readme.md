Text classification results
============

How to run
----------



#### Train the model:
```python
python -m examples.pytorch.text_classification.run_text_classifier -json_config examples/pytorch/text_classification/config/trec/XYZ.json
```

#### Run inference advance:
```python
python -m examples.pytorch.text_classification.inference_advance -json_config examples/pytorch/text_classification/config/trec/XYZ.json
```

#### Run inference:
```python
python -m examples.pytorch.text_classification.inference -json_config examples/pytorch/text_classification/config/trec/XYZ.json
```





TREC Results
-------


| GraphType\GNN  |   GAT-BiSep   |   GAT-BiFuse  |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-BiSep   | GGNN-BiFuse   | RGCN  | 
| -------------  | ------------- | --------------| ------------------- | -----------------  |-------------- | ------------- | ----- |
| Dependency     |     0.9480    |   0.9460      |         0.942       |      0.958         |      0.954    |     0.944    | 0.946 |
| Constituency   |     0.9420    |   0.9300      |         0.952       |      0.950         |      0.952    |     0.94    |  N/A  |
| NodeEmb        |      N/A      |    N/A        |         0.930       |      0.908         |       N/A     |     N/A      | N/A    |
| NodeEmbRefined |      N/A      |    N/A        |         0.940       |      0.926         |       N/A     |     N/A     |  N/A  |



CAirline Results
-------


| GraphType\GNN  |  GAT-BiSep   |  GGNN-BiSep   |GraphSage-BiSep|   RGCN        |
| -------------- | ------------ | ------------- |---------------|---------------|
| Dependency     | 0.7496       | 0.8020        | 0.7977        |    0.7525     |
| Constituency   | 0.7846       | 0.7933        | 0.7948        |    N/A        |
| NodeEmb        | N/A          | 0.8108        | 0.8108        |    N/A        | 
| NodeEmbRefined | N/A          | 0.7991        | 0.8020        |    N/A        |

