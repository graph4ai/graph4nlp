## fairseq
wmt 14
| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | BLEU|
| Transformer(beam=1)   | 64.5 | 37.7 | 23.9 | 15.5 | 30.09 | 
| Transformer(beam=3)   | 65.7 | 39.1 | 25.1 | 16.5 | 31.18 |
| lstm(beam=1)          | 57.9 | 30.2 | 17.6 | 10.5 | 23.84 |  4758MiB / 24268MiB
| lstm(beam=3)          | 60.6 | 33.1 | 19.8 | 12.2 | 25.87 |
| lstm w/o attn(beam=1) | 41.4 | 14.1 | 5.7  | 2.5  | 9.54  |
| lstm w/o attn(beam=3) | 42.2 | 14.9 | 6.3  | 2.9  | 10.37 |

``bash Train transformer
CUDA_VISIBLE_DEVICES=3,4,5,6 fairseq-train ~/software/fairseq/data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en  --max-tokens 4096 --max-update 30000         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0007         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --save-interval-updates 1000
``

``bash Train lstm based
CUDA_VISIBLE_DEVICES=5,6 fairseq-train ~/software/fairseq/data-bin/iwslt14.tokenized.de-en --arch lstm_wiseman_iwslt_de_en  --max-tokens 4096 --max-update 30000         --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0007         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --save-interval-updates 1000
``

``bash evaluation
fairseq-generate ~/software/fairseq/data-bin/iwslt14.tokenized.de-en   --path checkpoints_lstm/checkpoint_best.pt   --beam 1 --remove-bpe
``
