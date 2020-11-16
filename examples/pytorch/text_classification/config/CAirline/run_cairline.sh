#!/usr/bin/env bash

#python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/gat_bi_sep_dependency.yaml
#python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/gat_bi_sep_constituency.yaml

python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/ggnn_bi_sep_dependency.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/ggnn_bi_sep_constituency.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/ggnn_bi_sep_node_emb.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/ggnn_bi_sep_node_emb_refined_dependency.yaml

python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/graphsage_bi_sep_dependency.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/graphsage_bi_sep_constituency.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/graphsage_bi_sep_node_emb.yaml
python -m examples.pytorch.text_classification.run_text_classifier -config examples/pytorch/text_classification/config/CAirline/graphsage_bi_sep_node_emb_refined_dependency.yaml