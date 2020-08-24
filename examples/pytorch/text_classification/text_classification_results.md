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
Dependency graph:

GAT-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --gpu 0 --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option undirected --gnn gat
```

GAT-BiSep
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --gpu 1 --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_sep --gnn gat
```

GAT-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --gpu 1 --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_fuse --gnn gat
```


GGNN-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --direction_option undirected --gnn ggnn --gpu 0
```


GraphSAGE-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option undirected --gnn graphsage --gpu 0
```


Constituency graph:

GAT-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --gpu 0 --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option undirected --gnn gat
```


GAT-BiSep
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --gpu 0 --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_sep --gnn gat
```

GAT-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --gpu 0 --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_fuse --gnn gat
```



GraphSAGE-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option undirected --gnn graphsage --gpu 0
```

GraphSAGE-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option bi_fuse --gnn graphsage --gpu 0
```

GraphSAGE-BiSep
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option bi_sep --gnn graphsage --gpu 0
```

GGNN-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --direction_option undirected --gnn ggnn --gpu 1
```


<!-- IE graph:

GAT-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy none --graph_pooling avg_pool --graph_type ie --gpu 1 --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option undirected --gnn gat
```


GAT-BiSep
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy none --graph_pooling avg_pool --graph_type ie --gpu 1 --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_sep --gnn gat
```

GAT-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy none --graph_pooling avg_pool --graph_type ie --gpu 1 --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_fuse --gnn gat
```
 -->

TREC Results
-------

| GraphType\GNN  |  GAT-Undirected   |  GAT-BiSep    | GAT-BiFuse   | GraphSAGE-Undirected   |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-Undirected   |  GGNN-BiSep    | GGNN-BiFuse   | 
| ------------- |  -------------| ------------- |  -------------|  ------------- | ------------- |  -------------| ------------- | -------------  | ------------- |  
| Dependency     | 0.934  | 0.926  | 0.918 | 0.946 | 0.944 |  0.934  | 0.934 | 0.934 |  0.914 |
| Constituency | 0.922  | 0.942 | 0.948 | 0.942 |0.944 | 0.942  | 0.934 | 0.924 |  0.934 |

Dependency:
GAT-Undirected: 0.934
GAT-BiSep: 0.926
GAT-BiFuse: 0.918

GraphSAGE-Undirected: 0.946
GraphSAGE-BiSep: 0.944
GraphSAGE-BiFuse: 0.934

GGNN-Undirected: 0.934
GGNN-BiSep: 0.934
GGNN-BiFuse: 0.914

Constituency:
seq_info_encode_strategy: bilstm 
GAT-Undirected: 0.922
GAT-BiSep: 0.942
GAT-BiFuse: 0.948

GraphSAGE-Undirected: 0.942
GraphSAGE-BiSep: 0.944
GraphSAGE-BiFuse: 0.942

GGNN-Undirected: 0.934
GGNN-BiSep: 0.924
GGNN-BiFuse: 0.934






seq_info_encode_strategy: none
GAT-Undirected: 0.582
GAT-BiSep: 0.666
GAT-BiFuse: 0.796




