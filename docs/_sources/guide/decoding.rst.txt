Chapter 5. Decoder
===========================
.. image:: https://img.shields.io/github/forks/graph4ai/graph4nlp?style=social
        :target: https://github.com/graph4ai/graph4nlp/fork
.. image:: https://img.shields.io/github/stars/graph4ai/graph4nlp?style=social
        :target: https://github.com/graph4ai/graph4nlp
        
Graph4NLP implements two standard decoder modules for: sequence generation and tree generation, respectively. Generally, they take the ``GraphData`` encoded by GraphEncoder as inputs and generate target objects.

Roadmap
-----------------
The chapter starts with sections for decoding graphs with different strategy:

* :ref:`std-rnn-decoder`
* :ref:`std-tree-decoder`

.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:

   decoding/stdrnndecoder
   decoding/stdtreedecoder

    