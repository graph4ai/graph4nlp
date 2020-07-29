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
python -m graph4nlp.pytorch.test.graph_construction.run_dynamic_graph_construction --dataset=cora --gpu=0 --early-stop --gl-metric-type weighted_cosine --gl-epsilon 0.3 --gl-type node_emb
```
```bash
python -m graph4nlp.pytorch.test.graph_construction.run_dynamic_graph_construction --dataset=cora --gpu=0 --early-stop --gl-metric-type weighted_cosine --gl-epsilon 0.3 --gl-type node_emb_refined --init-adj-alpha 0.8 
```

Results with 2-layer GCN
-------

| Dataset  |    Raw graph    |  NodeEmb-based Graph   | Raw graph + NodeEmb-based Graph   |
| -------- | ------------- | ------------- | ------------- |
| Cora     | 82.02 (0.57)  | 58.48 (2.40) | 82.04 (0.52) |

<!-- NodeEmb-based Graph:
1 head attention, top-k 200: 27.14 (1.02)
1 head weighted cosine, top-k 10:  46.24 (2.63)
1 head weighted cosine, top-k 100: 50.58 (2.08)
4 heads weighted cosine, top-k 100: 49.52 (2.31)
1 head weighted cosine, epsilon 0.5: 50.78 (1.85)
1 head weighted cosine, epsilon 0.3: 58.48 (2.40)
2 head weighted cosine, epsilon 0.3: 56.80 (1.74)
4 head weighted cosine, epsilon 0.3: 57.62 (0.75) 
1 head weighted cosine, epsilon 0.3, smooth 0.5: 57.30 (1.32)

Raw graph + NodeEmb-based Graph:
1 head weighted cosine, epsilon 0.3, init-adj-alpha 0.8: 82.04 (0.52)
1 head weighted cosine, epsilon 0.3, init-adj-alpha 0.8, smoothness 0.1: 81.34 (1.15)
1 head weighted cosine, epsilon 0.3, init-adj-alpha 0.8, smoothness 0.2: 81.36 (0.77)
 -->



* All the accuracy numbers are averaged after 5 random runs.



TODO
-------

* Implement NodeEdgeEmbeddingBasedGraphConstruction


