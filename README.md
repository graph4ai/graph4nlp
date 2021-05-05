<p align="center">
<img src="./imgs/logo_graph4nlp.png" width="500" class="center" alt="logo"/>
    <br/>
</p>


[license-image]:https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license-url]:https://github.com/hugochan/IDGL/blob/master/LICENSE

[contributor-image]:https://img.shields.io/github/contributors/hugochan/IDGL
[contributor-url]:https://github.com/hugochan/IDGL/contributors

[contributing-image]:https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]:to_be_add

[issues-image]:https://img.shields.io/github/issues/hugochan/IDGL
[issues-url]:https://github.com/hugochan/IDGL/issues

[forks-image]:https://img.shields.io/github/forks/hugochan/IDGL?style=social
[forks-url]:https://github.com/hugochan/IDGL/fork

[stars-image]:https://img.shields.io/github/stars/hugochan/IDGL?style=social
[stars-url]:https://github.com/hugochan/IDGL/star

### for repo

![Last Commit](https://img.shields.io/github/last-commit/hugochan/IDGL)
[![Contributors][contributor-image]][contributor-url]
[![Contributing][contributing-image]][contributing-url]
[![License][license-image]][license-url]
[![Issues][issues-image]][issues-url]

### for docs
[![Fork][forks-image]][forks-url]
[![Star][stars-image]][stars-url]

# Graph4NLP

***Graph4NLP*** is the library for the easy use of Graph Neural Networks for NLP.

# Quick tour

***Graph4nlp*** aims to make it incredibly easy to use GNNs in all kinds of common NLP tasks (see [here](http://saizhuo.wang/g4nlp/index.html) for tutorial). Here is a simple example of how to use the [*Graph2seq*](http://saizhuo.wang/g4nlp/index.html) model (which is widely used in machine translation, question answering, semantic parsing, and various other nlp tasks that can be abstracted as graph to sequence problem and show superior performance). 

We also offer many other effective models such as graph classification models, graph to tree models, etc. If you are interested in related research problems, welcome to use our library and refer to our [graph4nlp survey](to_be_add).

```python
from graph4nlp.pytorch.modules.config import get_basic_args

opt = get_basic_args(graph_construction_name="node_emb", graph_embedding_name="gat", decoder_name="stdrnn")
graph2seq = Graph2Seq.from_args(opt=opt, vocab_model=vocab_model, device=torch.device("cuda:0"))

graph_list = [GraphData() for _ in range(2)]
tgt_seq = torch.Tensor([[1, 2, 3], [4, 5, 6]])
seq_out, _, _ = graph2seq(graph_list=graph_list, tgt_seq=tgt_seq)
print(seq_out.shape) 
```

# Overview

# Usage

# Models architectures

# More resources

# Citation