Dynamic Graph Construction
============



Dependencies
------------
- torch v1.0: the autograd support for sparse mm is only available in v1.0.
- requests
- sklearn

```bash
pip install torch==1.1.0 requests dgl
```

How to run
----------

Run with following:

#### Cora

```bash
python -m graph4nlp.pytorch.test.graph_construction.run_dynamic_graph_construction --dataset=cora --gpu=0 --early-stop --gl-topk 200  --gl-type node_emb
```
```bash
python -m graph4nlp.pytorch.test.graph_construction.run_dynamic_graph_construction --dataset=cora --gpu=0 --early-stop --gl-topk 100 --init-adj-alpha 0.85 --gl-type node_emb_refined
```

Results
-------

| Dataset  |    Raw graph    |  NodeEmb-based Graph   | Raw graph + NodeEmb-based Graph   |
| -------- | ------------- | ------------- | ------------- |
| Cora     |   | 26.98 (0.95) |   |
 
topk 100, alpha 0.85, mean 0.8018, std 0.0148



* All the accuracy numbers are averaged after 5 random runs.



TODO
-------


