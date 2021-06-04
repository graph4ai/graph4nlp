Requirements
---------
```bash
    pip install tensorboard
```




How to run
----------

#### Start the StanfordCoreNLP server for data preprocessing:

1) Download StanfordCoreNLP `https://stanfordnlp.github.io/CoreNLP/`
2) Go to the root folder and start the server:

```java
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```


#### run script
``` bash
python examples/pytorch/nmt/main.py --name test
```

NMT Results
-------

| Model  |  BLEU@1  |  BLEU@2  |BLEU@3| BLEU@4 | 
| -------------- | ------------ | ------------- |---------------|-----------|
| Dynamic + GCN     | 0.6104       | 0.4830         | 0.3920        | 0.3212|

