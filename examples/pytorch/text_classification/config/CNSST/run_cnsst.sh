#!/usr/bin/env bash

python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/gat_bi_sep_dependency.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/gat_bi_sep_constituency.yaml

python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/ggnn_bi_sep_dependency.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/ggnn_bi_sep_constituency.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/ggnn_bi_sep_node_emb.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/ggnn_bi_sep_node_emb_refined_dependency.yaml

python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/graphsage_bi_sep_dependency.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/graphsage_bi_sep_constituency.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/graphsage_bi_sep_node_emb.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CNSST/graphsage_bi_sep_node_emb_refined_dependency.yaml