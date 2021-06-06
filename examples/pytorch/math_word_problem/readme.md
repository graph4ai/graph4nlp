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
