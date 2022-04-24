# Graph2Tree for semantic parsing (take the jobs as example)

## Setup

### Start the StanfordCoreNLP server for data preprocessing

1) Download StanfordCoreNLP `https://stanfordnlp.github.io/CoreNLP/`
2) Go to the root folder and start the server:

```java
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

### Run with following

```python
python examples/pytorch/semantic_parsing/graph2tree/jobs/src/runner.py
```

## Results (Execution accuracy for jobs640)

| SAGE |undirected |  
| ---- | ---- |  
| Constituency | 91.4 |  
| Dynamic_node_emb | 92.1 |  




