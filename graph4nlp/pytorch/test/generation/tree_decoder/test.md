# Test instruction for tree decoder  

This file aims to illustrate how this tree decoder should be tested.

This tree decoder support:

- tree object decoding  

- parent feeding and sibling feeding  

- separate attention on different encoder and node type  

- copy and coverage attention  

- pretrained word embedding like GloVe and BERT.

## Some experimental details

1. add \</s\> and \<s\> or not in testing?  

2. use batching padding or not in testing?  

3. in sequence encoder, fuse h_s and c_s with fc layer or not?  

