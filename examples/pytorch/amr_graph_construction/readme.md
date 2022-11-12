# Graph2Tree for math word problem (MWP) For AMR graph and RGCN

## Setup

### Install and set

install the amrlib and the fast_align, and set the environment variables as follows
```bash
export FABIN_DIR=path_to_fast_align
export TOKENIZERS_PARALLELISM=True
```

### Run with following

#### Run with amr graph adn RGCN
```python
python examples/pytorch/math_word_problem/mawps/src_for_amr/runner.py -json examples/pytorch/math_word_problem/mawps/config_for_amr/dynamic_amr_undirected.json
```
#### Run with dependency graph and RGCN
```python
python examples/pytorch/math_word_problem/mawps/src_for_amr/runner.py -json examples/pytorch/math_word_problem/mawps/config_for_amr/dynamic_dependency_undirected.json
```

