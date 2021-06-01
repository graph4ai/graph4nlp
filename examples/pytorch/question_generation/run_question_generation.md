Question generation results
============

- Paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)



How to run
----------

#### Start the StanfordCoreNLP server for data preprocessing:

1) Download StanfordCoreNLP `https://stanfordnlp.github.io/CoreNLP/`
2) Go to the root folder and start the server:

```java
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```


#### Run the model:
```python
 python examples/pytorch/question_generation/run_question_generation_iclr.py   -task_config examples/pytorch/question_generation/config/squad_split2/qg.yaml  -g2s_config examples/pytorch/question_generation/config/squad_split2/XYZ.yaml
```

Note: 
1) `XYZ.yaml` should be replaced by the exact g2s config file such as `new_dependency_ggnn.yaml`.
2) You can find the output files in the `out/squad_split2/` folder. 
<!-- 3) You can save your time by downloading the preprocessed data for dependency graph from [here](https://drive.google.com/drive/folders/1UPrlBvzXXgmUqx41CzO6ULrA3E1v24P9?usp=sharing), and moving the `squad_split2` folder to `examples/pytorch/question_generation/data/`. -->

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


Test examples: 8964 | Time: 1590.95s |  Test scores: BLEU_1 = 0.39766, BLEU_2 = 0.24274, BLEU_3 = 0.16912, BLEU_4 = 0.12328, METEOR = 0.16371, ROUGE = 0.39663




Test examples: 8964 | Time: 83.54s |  Test scores: BLEU_1 = 0.39332, BLEU_2 = 0.22917, BLEU_3 = 0.15234, BLEU_4 = 0.10504, METEOR = 0.15556, ROUGE = 0.38295 coverage



