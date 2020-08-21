Text classification results
============

- GAT paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- GAT-BiSep paper link: [https://arxiv.org/abs/1808.07624](https://arxiv.org/abs/1808.07624)
- GAT-BiFuse paper link: [https://arxiv.org/abs/1908.04942](https://arxiv.org/abs/1908.04942)

- Convolutional Neural Networks for Sentence Classification: [https://arxiv.org/abs/1408.5882](https://arxiv.org/abs/1408.5882)

- TREC: [https://cogcomp.seas.upenn.edu/Data/QA/QC/](https://cogcomp.seas.upenn.edu/Data/QA/QC/)



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

```java
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

GAT
```python
python -m pdb -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --direction_option uni --graph_type dependency --gpu 3 --num_heads 8 --num_out_heads 1 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.2


```

TREC Results
-------

| GraphType\GNN  |  GAT-Undirected   |  GAT-BiSep    | GAT-BiFuse   | GraphSAGE-Undirected   |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-Undirected   |  GGNN-BiSep    | GGNN-BiFuse   | 
| ------------- |  -------------| ------------- |  -------------|  ------------- | ------------- |  -------------| ------------- | -------------  | ------------- |  
| Dependency     |   |   |  |  |  |   |  |  |  
| Constituency     |   |   |  |  |  |   |  |  |  
| IE     |   |   |  |  |  |   |  |  |  


fix word_emb
bs: 50
avg_pool


GAT-Undirected : 0.906





TODO
-------

- early stopping, loading best model, pretrianed glove vectors, hyper-param tuning, ggnn/graphsage, seq_info_encode_strategy (check node ordering)

