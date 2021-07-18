.. _guide-customize:

Customizing your own dataset
===========
The first thing to know about Graph4NLP's Dataset class is that the basic element for a dataset is ``DataItem``, which
can be arbitrary collection of data instances, including natural language sentences (string), token list (list) or graphs.

Customizing DataItem
----------
The base class for data item is ``DataItem``. ``DataItem`` has an abstract method ``extract()``, which returns the input
and output tokens. To create your own ``DataItem`` class, simply inherit the base class and implement the ``extract()``
method.

Customizing downloading
----------
Downloading can be decomposed into 2 steps: 1) check whether file exist and 2) download the missing files.
To customize checking, the file names must be specified:

.. code-block::

    @property
    def raw_file_names(self) -> dict:
        raise NotImplementedError


The file names will be concatenated with ``self.raw_dir`` to compose the complete file path. To customize downloading,
simply override the ``download()`` method, since the root downloading method in the base class ``Dataset`` is defined
in such a way.

.. code-block::

    class Dataset:
        def _download(self):
            if all([os.path.exists(raw_path) for raw_path in self.raw_file_paths.values()]):
                return

            os.makedirs(self.raw_dir, exist_ok=True)
            self.download()

        @abc.abstractmethod
        def download(self):
            """Download the raw data from the Internet."""
            raise NotImplementedError


Customizing processing
----------
Similar to the way we customize downloading, processing consists of the same set of sub-steps. Except for an additional
check for split ratio.

.. code-block::

    def _process(self):
        if all([os.path.exists(processed_path) for processed_path in self.processed_file_paths.values()]):
            if 'val_split_ratio' in self.__dict__:
                UserWarning(
                    "Loading existing processed files on disk. Your `val_split_ratio` might not work since the data have"
                    "already been split.")
            return
        if self.for_inference and \
                all([(os.path.exists(processed_path) or self.processed_file_names['data'] not in processed_path) for
                     processed_path in self.processed_file_paths.values()]):
            return

        os.makedirs(self.processed_dir, exist_ok=True)

        self.read_raw_data()

        if self.for_inference:
            self.test = self.build_topology(self.test)
            self.vectorization(self.test)
            data_to_save = {'test': self.test}
            torch.save(data_to_save, self.processed_file_paths['data'])
        else:
            self.train = self.build_topology(self.train)
            self.test = self.build_topology(self.test)
            if 'val' in self.__dict__:
                self.val = self.build_topology(self.val)

            self.build_vocab()

            self.vectorization(self.train)
            self.vectorization(self.test)
            if 'val' in self.__dict__:
                self.vectorization(self.val)

            data_to_save = {'train': self.train, 'test': self.test}
            if 'val' in self.__dict__:
                data_to_save['val'] = self.val
            torch.save(data_to_save, self.processed_file_paths['data'])

            vocab_to_save = self.vocab_model
            torch.save(vocab_to_save, self.processed_file_paths['vocab'])


Customizing batching
-----------
The runtime iteration over dataset is performed by PyTorch's dataloader. And since the basic composing element is
``DataItem``, it is our job to convert the low-level list of ``DataItem`` fetched by ``torch.DataLoader`` to the batch
data we want.
``Dataset.collate_fn()`` is designed to do this job.

.. code-block::

    @staticmethod
    @abc.abstractmethod
    def collate_fn(data_list):
        """Takes a list of data and convert it to a batch of data."""
        raise NotImplementedError

It takes in a list of DataItem and returns the expected type of data required by the model. Interested readers may
refer to the examples we provided in the source code for practical usages.
