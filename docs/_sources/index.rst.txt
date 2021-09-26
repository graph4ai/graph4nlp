.. Graph4NLP documentation master file, created by
   sphinx-quickstart on Tue Aug 25 16:43:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Graph4NLP Documentation
=====================================
.. image:: https://img.shields.io/github/forks/graph4ai/graph4nlp?style=social
        :target: https://github.com/graph4ai/graph4nlp/fork
.. image:: https://img.shields.io/github/stars/graph4ai/graph4nlp?style=social
        :target: https://github.com/graph4ai/graph4nlp

**Graph4NLP** is an easy-to-use library for R&D at the intersection of **Deep Learning on Graphs** and
**Natural Language Processing** (i.e., DLG4NLP). It provides both **full implementations** of state-of-the-art models for
data scientists and also **flexible interfaces** to build customized models for researchers and developers with whole-pipeline
support. Built upon highly-optimized runtime libraries including `DGL <https://github.com/dmlc/dgl>`_ , **Graph4NLP** has both
high running efficiency and great extensibility. The architecture of **Graph4NLP** is shown in the following figure, where boxes
with dashed lines represents the features under development.

This library has the following key features:

1. **Easy-to-use and Flexible:** Provides both full implementations of state-of-the-art models and alsoflexible interfaces to build customized models with whole-pipeline support.
2. **Rich Set of Learning Resources:** Provide a variety of learning materials including code demos, code documentations, research tutorials and videos, and paper survey.
3. **High Running Efficiency and Extensibility:** Build upon highly-optimized runtime libraries including DGL and provide highly modularization blocks.
4. **Comprehensive Code Examples:** Provide a comprehensive collection of NLP applications and the corresponding code examples for quick-start.

Graph4NLP consists of four different layers: 1) Data Layer, 2) Module Layer, 3) Model Layer,
and 4) Application Layer, as illustrated in the following figure.

.. figure:: ../arch.png
   :align: center
   :width: 600

   Graph4NLP Architecture

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Get Started
   :hidden:

   welcome/installation


.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   guide/graphdata
   guide/dataset
   guide/construction
   guide/gnn
   guide/decoding
   guide/classification
   guide/evaluation


.. toctree::
   :glob:
   :maxdepth: 4
   :hidden:
   :caption: Module API references

   modules/data
   modules/datasets
   modules/graph_construction
   modules/graph_embedding
   modules/prediction
   modules/loss
   modules/evaluation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorial/text_classification
   tutorial/semantic_parsing
   tutorial/math_word_problem
   tutorial/knowledge_graph_completion



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
