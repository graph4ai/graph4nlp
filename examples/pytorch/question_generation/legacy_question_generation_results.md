Question generation results
============

- Paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)


Dependencies
------------

Install the required python packages:

```bash
pip install -r examples/pytorch/question_generation/requirements.txt
```


How to run
----------

#### Start the StanfordCoreNLP server for data preprocessing:

1) Download StanfordCoreNLP `https://stanfordnlp.github.io/CoreNLP/`
2) Go to the root folder and start the server:

```java
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

#### Run the model with grid search:

```python
    python -m examples.pytorch.question_generation.run_question_generation -config examples/pytorch/question_generation/config/squad_split2/XYZ.yaml --grid_search
```

Note: 
1) `XYZ.yaml` should be replaced by the exact config file.
2) You can find the output files in the `out/squad_split2/` folder. 
3) You can save your time by downloading the preprocessed data for dependency graph from [here](https://drive.google.com/drive/folders/1UPrlBvzXXgmUqx41CzO6ULrA3E1v24P9?usp=sharing), and moving the `squad_split2` folder to `examples/pytorch/question_generation/data/`.

<!-- 
SQuAD-split2 Results
-------

| GraphType\GNN  |  GAT-Undirected   |  GAT-BiSep    | GAT-BiFuse   | GraphSAGE-Undirected   |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-Undirected   |  GGNN-BiSep    | GGNN-BiFuse   | 
| ------------- |  -------------| ------------- |  -------------|  ------------- | ------------- |  -------------| ------------- | -------------  | ------------- |  
| Dependency     |  |   |  | | |    |  |  |   |
| Constituency (word & non-word nodes) |  |  | |  | |  | ||   |
| NodeEmb | N/A  | N/A | N/A | | - | -  |  | - |  - |
| NodeEmbRefined (line) | N/A  | N/A | N/A |  |- |   -|  | - | -  |
| NodeEmbRefined (dependency) | N/A  | N/A | N/A | |- |   -|  | - | -  |
| NodeEmbRefined (constituency) | N/A  | N/A | N/A |  |- |   -|  | - | -  |
 -->



