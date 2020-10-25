Text classification results
============

How to run
----------

```python
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/graphsage_bi_sep_node_emb.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/gat_bi_sep_dependency.yaml
```

Run the model with grid search:

```python
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/trec_finetune/XYZ.yaml --grid_search
```


TREC Results
-------

| GraphType\GNN  |  GAT-Undirected   |  GAT-BiSep    | GAT-BiFuse   | GraphSAGE-Undirected   |  GraphSAGE-BiSep    | GraphSAGE-BiFuse   |  GGNN-Undirected   |  GGNN-BiSep    | GGNN-BiFuse   | 
| ------------- |  -------------| ------------- |  -------------|  ------------- | ------------- |  -------------| ------------- | -------------  | ------------- |  
| Dependency     | 0.934*  | 0.948*  | 0.950* | 0.946* | 0.944* |  0.942  | 0.934 | 0.946* |  0.938* |
| Constituency (word & non-word nodes) | 0.932*  | 0.942* | 0.938* | 0.934 |0.928 | 0.944*  | 0.920 |0.944* |  0.940* |
| NodeEmb | N/A  | N/A | N/A | 0.936 | 0.932 | 0.928  |  | | |
| NodeEmbRefined (dependency) | N/A  | N/A | N/A |0.928 |0.928 | 0.930  |  |  |   |
<!-- | NodeEmbRefined (constituency) | N/A  | N/A | N/A |  | |   |  |  |  | -->
<!-- | Constituency | 0.922  | 0.942 | 0.948 | 0.942 |0.944 | 0.946  | 0.934 | 0.924 |  0.934 | -->
<!-- | NodeEmbRefined (line) | N/A  | N/A | N/A | 0.936 |- |   -|  | - | -  | -->


CAirline Results
-------

| GraphType\GNN  |  GAT-BiSep   |  GGNN-BiSep   |GraphSage-BiSep| 
| -------------- | ------------ | ------------- |---------------|
| Dependency     | 0.8016       | 0.7685        | 0.7622        |
| Constituency   | 0.7669       | 0.7732        | 0.7638        |
| NodeEmb        | N/A          | 0.7748        | 0.7591        |
| NodeEmbRefined | N/A          | 0.7654        | 0.7780        |


Runtime (train one epoch/test all)
| GraphType\GNN  |   GAT-BiSep  |  GGNN-BiSep   |GraphSage-BiSep|
| -------------- | -------------| ------------- |---------------|
| Dependency     |  9.59s/1.89s |  9.48s/1.92s  |  9.43s/2.03s  |
| Constituency   | 13.45s/2.57s | 13.20s/2.64s  | 12.70s/2.72s  |
| NodeEmb        | N/A          | 15.00s/2.98s  | 16.64s/3.46s  |
| NodeEmbRefined | N/A          | 16.76s/3.50s  | 16.04s/3.49s  |


CNSST Results
-------

| GraphType\GNN  |  GAT-BiSep   |  GGNN-BiSep   |GraphSage-BiSep| 
| -------------- | ------------ | ------------- |---------------|
| Dependency     | 0.5381       | 0.5112        | 0.5031        |
| Constituency   | 0.5399       | 0.5085        | 0.5345        |
| NodeEmb        | N/A          | 0.5085        | 0.4987        |
| NodeEmbRefined | N/A          | 0.5157        | 0.5265        |


Runtime (train one epoch/test all)
| GraphType\GNN  |   GAT-BiSep  |  GGNN-BiSep   |GraphSage-BiSep|
| -------------- | -------------| ------------- |---------------|
| Dependency     | 15.93s/6.12s |   8.43s/2.88s |   8.23s/2.93s |
| Constituency   | 17.42s/7.58s |  10.40s/3.99s |  10.99s/4.10s |
| NodeEmb        | N/A          |  13.73s/5.06s |  15.98s/4.67s |
| NodeEmbRefined | N/A          |  15.40s/5.81s |  12.65s/5.26s |


Note: 
-------
- results denoted with '*' were hyperparameter fine-tuned with grid search extensively.








<!-- Dependency graph:

GAT-Undirected
```python
python -m pdb -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option undirected --gnn gat
```

GAT-BiSep
```python
python -m pdb -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_sep --gnn gat --num_layers 1
```

GAT-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_fuse --gnn gat --num_layers 1
```



GraphSAGE-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option undirected --gnn graphsage
```

GraphSAGE-BiSep
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option bi_sep --gnn graphsage --num_layers 1
```


GraphSAGE-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.2 --graphsage_aggreagte_type lstm --direction_option bi_fuse --gnn graphsage --num_layers 1
```



GGNN-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --direction_option undirected --gnn ggnn
```

GGNN-BiSep
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --direction_option bi_sep --gnn ggnn --num_layers 1
```


GGNN-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type dependency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --direction_option bi_fuse --gnn ggnn --num_layers 1
```


Constituency graph:

GAT-Undirected
```python
python -m pdb -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option undirected --gnn gat
```


GAT-BiSep
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_sep --gnn gat --num_layers 1
```

GAT-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_fuse --gnn gat --num_layers 1
```



GraphSAGE-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option undirected --gnn graphsage
```

GraphSAGE-BiSep
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option bi_sep --gnn graphsage --num_layers 1
```


GraphSAGE-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option bi_fuse --gnn graphsage --num_layers 1
```



GGNN-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --direction_option undirected --gnn ggnn
```

GGNN-BiSep
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --direction_option bi_sep --gnn ggnn
```

GGNN-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --graph_type constituency --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --direction_option bi_fuse --gnn ggnn
```


Node embedding based dynamic graph:

GGNN-Undirected
```python
python -m pdb -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --direction_option undirected --gnn ggnn --graph_type node_emb --gl_num_heads 1 --gl_epsilon 0.7
```


GraphSAGE-Undirected
```python
python -m pdb -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option undirected --gnn graphsage --graph_type node_emb --gl_num_heads 4 --gl_epsilon 0.5 --gl_smoothness_ratio 0.1 --gl_connectivity_ratio 0. --gl_sparsity_ratio 0.  --gpu 1 
```




Node embedding based refined dynamic graph:

```python
python -m pdb -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option undirected --gnn graphsage --graph_type node_emb_refined --init_graph_type line --gl_num_heads 1 --gl_epsilon 0.7 --gpu 1 --init_adj_alpha 0.2
```


```python
python -m pdb -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option undirected --gnn graphsage --graph_type node_emb_refined --init_graph_type dependency --gl_num_heads 1 --gl_epsilon 0.5 --init_adj_alpha 0.2
```


```python
python -m pdb -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy bilstm --graph_pooling avg_pool --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.1 --graphsage_aggreagte_type lstm --direction_option undirected --gnn graphsage --graph_type node_emb_refined --init_graph_type constituency --gl_num_heads 1 --gl_epsilon 0.5 --gpu 1 --init_adj_alpha 0.2
```

graphsage undirected, init_graph_type dependency, gl_epsilon 0.5, init_adj_alpha 0.2: 0.924


graphsage undirected, init_graph_type line, gl_epsilon 0.7, init_adj_alpha 0.2: 0.924

graphsage undirected, init_graph_type dependency, gl_epsilon 0.5, init_adj_alpha 0.2, new_norm: 0.918

graphsage undirected, init_graph_type line, gl_epsilon 0.7, init_adj_alpha 0.2, new_norm: 0.918 -->




<!-- IE graph:

GAT-Undirected
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy none --graph_pooling avg_pool --graph_type ie --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option undirected --gnn gat
```


GAT-BiSep
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy none --graph_pooling avg_pool --graph_type ie --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_sep --gnn gat
```

GAT-BiFuse
```python
python -m examples.pytorch.text_classification.run_text_classifier --pre_word_emb_file ~/Research/Resource/glove-vectors/glove.840B.300d.txt --node_edge_emb_strategy mean --seq_info_encode_strategy none --graph_pooling avg_pool --graph_type ie --num_heads 1 --num_out_heads 2 --num_hidden 300 --word_drop 0.4 --rnn_drop 0.1 --gnn_drop 0.6 --gat_attn_drop 0.3 --direction_option bi_fuse --gnn gat
```
 -->




<!-- Dependency:
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


node_emb:

GraphSAGE-Undirected: 

epsilon 0.8, head 1: 0.928
epsilon 0.8, head 2: 0.914
epsilon 0.8, head 4: 0.934
epsilon 0.8, head 6: 0.926
epsilon 0.7, head 1: 0.924
epsilon 0.7, head 2: 0.920
epsilon 0.7, head 4: 0.932
epsilon 0.7, head 6: 0.936
epsilon 0.6, head 1: 0.932
epsilon 0.6, head 2: 0.922
epsilon 0.6, head 4: 0.936
epsilon 0.6, head 6: 0.930
epsilon 0.5, head 1: 0.932
epsilon 0.5, head 2: 0.924
epsilon 0.5, head 4: 0.928
epsilon 0.5, head 6: 0.926


epsilon 0.8, head 4, gl_smoothness_ratio 0.1, gl_connectivity_ratio 0.05, gl_sparsity_ratio 0.1: 0.934



node_emb_refined:

GraphSAGE-Undirected: 

epsilon 0.7, head 1, init_adj_aplha 0.2: 0.934
epsilon 0.7, head 1, init_adj_aplha 0.3: 0.936
epsilon 0.7, head 1, init_adj_aplha 0.4: 0.930


seq_info_encode_strategy: none
GAT-Undirected: 0.582
GAT-BiSep: 0.666
GAT-BiFuse: 0.796 -->
