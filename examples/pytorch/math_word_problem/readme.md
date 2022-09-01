# Graph2Tree for math word problem (MWP) (take the MAWPS dataset as example)

## Setup

### Start the StanfordCoreNLP server for data preprocessing

1) Download StanfordCoreNLP `https://stanfordnlp.github.io/CoreNLP/`
2) Go to the root folder and start the server:

```java
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

### Run with following

```python
python examples/pytorch/math_word_problem/mawps/src/runner.py
```

## Results (Solution accuracy for MAWPS)

| SAGE |undirected |  
| ---- | ---- |  
| Dynamic_node_emb | 76.4 | 

|  Graph  | rgcn  | gcn  |
| ---- | ---- | ---- |
| amr |  72.0 / 71.8 / 69.4 / 70.8(num_bases=3) / 73.6 / 76.4、70.8、72.8 | 71.2   | size=77
| dependency | 74.8 / 76.0/ 76.4(normal)76.0/  (5)77.6/78.0/75.2 | 76.4/74.8 | size=38
| constituency | 74.0 | 75.6 |

|  Graph  | rgcn1 | rgcn2 | rgcn3 | gcn1 | gcn2 | gcn3 |
| ---- | ---- | ---- | --- | --- | --- | --- |
| amr |  76.0(4) | 74.8(4) | 72.4(4) | 74.8 | 73.2 | 72.4 |
| dependency | 77.6(5) | 77.6(5) | 76.0(5) | 74.8 | 76.0 | 74.0 |
| dependency+edge_inf | | | |74.4 | 74.8 | |

python examples/pytorch/amr_graph_construction/runner.py --json_config=examples/pytorch/amr_graph_construction/mawps/config/dynamic_sage_undirected.json
