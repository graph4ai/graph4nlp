How to run
----------

+ Download Summarization dataset like CNN/DM to folder 'raw/' and use `preprocess.py` to preprocess the original dataset.
+ For quick tests, we only used 30,000 pieces of data from the CNN dataset and the processed files are saved in `raw/train_3w.json`, `raw/test.json` and `raw/val.json`.
+ Run with following:

```bash
python examples/pytorch/summarization/cnn/main.py -json_config examples/pytorch/summarization/cnn/config/sum_g2s.json
```

Note: 
1) `XYZ.yaml` should be replaced by the exact g2s config file such as `gcn_bifuse.yaml`.
2) You can find the output files in the `out/cnn/` folder. 

|  Dataset |    Model   | Graph Construction   | Evaluation | Performance |
| -------- | ---------- | ------------ | ------ |-------------|
| CNN (30k)|  GCN bi-fuse  |     Dependency     |  ROUGE-1  | 29.2        |
