Install Graph4NLP
******

Currently, users can install Graph4NLP via **pip** or **source code**. Graph4NLP supports the following OSes:

- Linux-based systems (tested on Ubuntu 18.04 and later)
- macOS (only CPU version)
- Windows 10 (only support pytorch >= 1.8)

Installation via pip (binaries)
===========

We provide pip wheels for all major OS/PyTorch/CUDA combinations. Note that we highly recommend `Windows` users refer to `Installation via source code` due to compatibility.

Ensure that at least PyTorch (>=1.6.0) is installed:
------------------

Note that `>=1.6.0` is ok.

.. code-block::

    $ python -c "import torch; print(torch.__version__)"
    >>> 1.6.0


Find the CUDA version PyTorch was installed with (for GPU users):
-------------
.. code-block::

    $ python -c "import torch; print(torch.version.cuda)"
    >>> 10.2


Install the relevant dependencies:
--------------

`torchtext` is needed since Graph4NLP relies on it to implement embeddings.
Please pay attention to the PyTorch requirements before installing `torchtext` with the following script! For detailed version matching please refer [here](https://pypi.org/project/torchtext/).

.. code-block::
    pip install torchtext # >=0.7.0



Install Graph4NLP
-----------

.. code-block::

    pip install graph4nlp${CUDA}


where `${CUDA}` should be replaced by the specific CUDA version (`none` (CPU version), `"-cu92"`, `"-cu101"`, `"-cu102"`, `"-cu110"`). The following table shows the concrete command lines. For CUDA 11.1 users, please refer to `Installation via source code`.

.. list-table:: Supported platforms
   :widths: 25 50
   :header-rows: 1

   * - Platform
     - Command
   * - CPU
     - `pip install graph4nlp`
   * - CUDA 9.2
     - `pip install graph4nlp-cu92`
   * - CUDA 10.1
     - `pip install graph4nlp-cu101`
   * - CUDA 10.2
     - `pip install graph4nlp-cu102`
   * - CUDA 11.0
     - `pip install graph4nlp-cu110`


Installation via source code
==============

Ensure that at least PyTorch (>=1.6.0) is installed:
------------------

Note that `>=1.6.0` is ok.

.. code-block::

    $ python -c "import torch; print(torch.__version__)"
    >>> 1.6.0


Find the CUDA version PyTorch was installed with (for GPU users):
-------------
.. code-block::

    $ python -c "import torch; print(torch.version.cuda)"
    >>> 10.2


Install the relevant dependencies:
--------------

`torchtext` is needed since Graph4NLP relies on it to implement embeddings.
Please pay attention to the PyTorch requirements before installing `torchtext` with the following script! For detailed version matching please refer [here](https://pypi.org/project/torchtext/).

.. code-block::
    pip install torchtext # >=0.7.0


Download the source code of `Graph4NLP` from Github:
--------------

.. code-block::

    git clone https://github.com/graph4ai/graph4nlp.git
    cd graph4nlp


Configure the CUDA version
--------------

Then run `./configure` (or `./configure.bat`  if you are using Windows 10) to config your installation. The configuration program will ask you to specify your CUDA version. If you do not have a GPU, please type 'cpu'.

.. code-block::

    ./configure


Install Graph4NLP
----------

Finally, install the package:

.. code-block::

    python setup.py install

Enjoy!
