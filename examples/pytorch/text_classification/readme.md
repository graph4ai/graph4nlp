Text classification results
============

How to run
----------

```python
python examples/pytorch/text_classification/run_text_classifier.py -config examples/pytorch/text_classification/config/trec/ggnn_bi_fuse_constituency.yaml
python examples/pytorch/text_classification/run_text_classifier.py -config examples/pytorch/text_classification/config/CAirline/gat_bi_sep_dependency.yaml
```

Run the model with grid search:

```python
python examples/pytorch/text_classification/run_text_classifier.py -config examples/pytorch/text_classification/config/trec_finetune/XYZ.yaml --grid_search
```


TREC Results
-------

| GraphType\GNN  |   GAT-BiSep   |   GAT-BiFuse  | GraphSAGE-Undirected |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-Undirected |  GGNN-BiSep   | GGNN-BiFuse   | 
| -------------  | ------------- | --------------| -------------------- | ------------------- | -----------------  | ---------------- | ------------- | ------------- |  
| Dependency     |     0.9500    |   0.9500      |       0.946          |         0.944       |      0.942         |      0.9340      |      0.9460   |     0.9380    |
| Constituency   |     0.9440    |   0.9380      |       0.934          |         0.928       |      0.944         |      0.9200      |      0.9480   |     0.9400    |
| NodeEmb        |      N/A      |    N/A        |       0.936          |         0.932       |      0.928         |                  |               |               |
| NodeEmbRefined |      N/A      |    N/A        |       0.928          |         0.928       |      0.930         |                  |               |               |



CAirline Results
-------

| GraphType\GNN  |  GAT-BiSep   |  GGNN-BiSep   |GraphSage-BiSep| 
| -------------- | ------------ | ------------- |---------------|
| Dependency     | 0.7875       | 0.8020        | 0.7962        |
| Constituency   | 0.7817       | 0.7948        | 0.7904        |
| NodeEmb        | N/A          | 0.8122        | 0.8195        |
| NodeEmbRefined | N/A          | 0.8020        | 0.8122        |

