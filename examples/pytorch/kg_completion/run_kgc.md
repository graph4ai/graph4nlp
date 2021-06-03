Installation 
===========
+ Download the default English model used by **spaCy**, which is installed in the previous step 
```bash
pip install spacy
python -m spacy download en_core_web_sm
```
+ Run the preprocessing script for WN18RR and Kinship: ```sh preprocess.sh```
+ You can now run the model

How to run
----------


#### Run the model:

If you run the task for the first time, run with:
```bash
python examples/pytorch/kg_completion/kinship/main.py --data kinship --model ggnn_distmult --preprocess
python -m examples.pytorch.kg_completion.WN18RR.main --data WN18RR --model gcn_distmult --lr 0.005 --preprocess
```
Then run:
```bash
python -m examples.pytorch.kg_completion.kinship.main --data kinship --model ggnn_distmult
python -m examples.pytorch.kg_completion.WN18RR.main --data WN18RR --model gcn_distmult --lr 0.005
```


Note: 
1) `XYZ.yaml` should be replaced by the exact g2s config file such as `new_dependency_ggnn.yaml`.
2) You can find the output files in the `out/squad_split2/` folder. 
<!-- 3) You can save your time by downloading the preprocessed data for dependency graph from [here](https://drive.google.com/drive/folders/1UPrlBvzXXgmUqx41CzO6ULrA3E1v24P9?usp=sharing), and moving the `squad_split2` folder to `examples/pytorch/question_generation/data/`. -->


SQuAD-split2 Dependency + GGNN results:
|     Method        | BLEU_1 | BLEU_2 | BLEU_3 | BLEU_4 | METEOR | ROUGE |
| ----------------- | ------ | ------ | ------ | ------ | ------ | ----- |
| Dependency + GGNN | 0.43297|0.28016 |0.20205 |0.15175 | 0.18994|0.43401|