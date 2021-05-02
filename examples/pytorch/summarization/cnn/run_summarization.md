How to run
----------

+ Download Summarization dataset like CNN/DM to folder 'raw/' and use `preprocess.py` to preprocess the original dataset.
+ For quick tests, we only used 30,000 pieces of data from the CNN dataset and the processed files are saved in `raw/train_3w.json`, `raw/test.json` and `raw/val.json`.
+ Run with following:

```python
python -m examples.pytorch.summarization.cnn.main -g2s_config examples/pytorch/summarization/cnn/config/gcn_bifuse.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml
```