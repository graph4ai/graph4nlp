## run script
``` bash
CUDA_VISIBLE_DEVICES=6 python examples/pytorch/nmt/main.py --name test
```

NMT Results
-------

| Model  |  BLEU@1  |  BLEU@2  |BLEU@3| BLEU@4 | 
| -------------- | ------------ | ------------- |---------------|-----------|
| Dynamic + GCN     | 0.6104       | 0.4830         | 0.3920        | 0.3212|

