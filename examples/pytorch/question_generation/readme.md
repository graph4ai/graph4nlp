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


#### Train the model:
```python
  python -m examples.pytorch.question_generation.main -task_config examples/pytorch/question_generation/config/squad_split2/qg.yaml  -g2s_config examples/pytorch/question_generation/config/squad_split2/XYZ.yaml
```

#### Run inference advance:
```python
  python -m examples.pytorch.question_generation.inference_advance -task_config examples/pytorch/question_generation/config/squad_split2/qg.yaml  -g2s_config examples/pytorch/question_generation/config/squad_split2/XYZ.yaml
```

#### Run inference:
```python
  python -m examples.pytorch.question_generation.inference -task_config examples/pytorch/question_generation/config/squad_split2/qg.yaml  -g2s_config examples/pytorch/question_generation/config/squad_split2/XYZ.yaml
```


Note: 
1) `XYZ.yaml` should be replaced by the exact g2s config file such as `new_dependency_ggnn.yaml`.
2) You can find the output files in the `out/squad_split2/` folder. 
<!-- 3) You can save your time by downloading the preprocessed data for dependency graph from [here](https://drive.google.com/drive/folders/1UPrlBvzXXgmUqx41CzO6ULrA3E1v24P9?usp=sharing), and moving the `squad_split2` folder to `examples/pytorch/question_generation/data/`. -->


SQuAD-split2 Dependency + GGNN results:
|     Method        | BLEU_1 | BLEU_2 | BLEU_3 | BLEU_4 | METEOR | ROUGE |
| ----------------- | ------ | ------ | ------ | ------ | ------ | ----- |
| Dependency + GGNN | 0.43297|0.28016 |0.20205 |0.15175 | 0.18994|0.43401|

