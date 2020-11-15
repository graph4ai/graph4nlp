.. Graph4NLP documentation master file, created by
   sphinx-quickstart on Tue Aug 25 16:43:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Graph4NLP Documentation
=====================================
Graph4NLP is a library utilizing deep learning on graphs to carry out natural language processing tasks. We believe
this library will boost research in relevant fields. This library has the following advantages:

- End-to-End supporting: Graph4NLP implements every necessary components in the pipeline of natural language processing tasks.

- Large coverage: Graph4NLP implements all popular methods in graph deep learning.

- To be added here.

Sphinx reference: `reStructuredText Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_

Sphinx themes: `Link <https://sphinx-themes.org/>`_

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
   :caption: Model Examples
   :hidden:

   tutorial/example1
   tutorial/example2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
