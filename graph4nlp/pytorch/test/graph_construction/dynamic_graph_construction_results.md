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
python -m graph4nlp.pytorch.test.graph_construction.run_dynamic_graph_construction --dataset=cora --gpu=0 --early-stop --gl-top-k 100  --gl-type node_emb
```
```bash
python -m graph4nlp.pytorch.test.graph_construction.run_dynamic_graph_construction --dataset=cora --gpu=0 --early-stop --gl-type node_emb_refined --init-adj-alpha 0.85 --gl-epsilon 0 --gl-num-heads 1
```

Results with 2-layer GCN
-------

| Dataset  |    Raw graph    |  NodeEmb-based Graph   | Raw graph + NodeEmb-based Graph   |
| -------- | ------------- | ------------- | ------------- |
| Cora     | 82.02 (0.57)  | 26.98 (0.95) | 80.36 (1.01) |

NodeEmb-based Graph:
1 head attention, top-k 200: 27.14 (1.02)
1 head weighted cosine, top-k 10:  46.24 (2.63)


* All the accuracy numbers are averaged after 5 random runs.



TODO
-------
* Explore other similarity metric functions.
* Explore graph regularization techniques.


