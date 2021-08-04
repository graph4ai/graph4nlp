.. _guide-customize:

Customizing your own dataset
===========
The first thing to know about Graph4NLP's Dataset class is that the basic element for a dataset is ``DataItem``, which
can be arbitrary collection of data instances, including natural language sentences (string), token list (list) or graphs.
In this guide we will use the JOBS dataset as an example to walk readers through the process of customizing a dataset.

Customizing DataItem
----------
The base class for data item is ``DataItem``. ``DataItem`` has an abstract method ``extract()``, which returns the input
and output tokens. To create your own ``DataItem`` class, simply inherit the base class and implement the ``extract()``
method.

The Jobs dataset inherits the ``Text2TextDataset`` base class, which uses ``Text2TextDataItem`` as its composing elements.
``Text2TextDataItem`` implements its own ``extract`` method which returns the list(s) of tokens contained in the text graph.

.. code-block::

    class Text2TextDataItem(DataItem):
        def __init__(self, input_text, output_text, tokenizer, share_vocab=True):
            super(Text2TextDataItem, self).__init__(input_text, tokenizer)
            self.output_text = output_text
            self.share_vocab = share_vocab
        def extract(self):
            g: GraphData = self.graph
            input_tokens = []
            for i in range(g.get_node_num()):
                if self.tokenizer is None:
                    tokenized_token = g.node_attributes[i]['token'].strip().split(' ')
                else:
                    tokenized_token = self.tokenizer(g.node_attributes[i]['token'])

                input_tokens.extend(tokenized_token)

            if self.tokenizer is None:
                output_tokens = self.output_text.strip().split(' ')
            else:
                output_tokens = self.tokenizer(self.output_text)

            if self.share_vocab:
                return input_tokens + output_tokens
            else:
                return input_tokens, output_tokens


    class JobsDataset(Text2TextDataset):
            def __init__(self, root_dir,
                 topology_builder, topology_subdir,
                #  pretrained_word_emb_file=None,
                 pretrained_word_emb_name="6B",
                 pretrained_word_emb_url=None,
                 pretrained_word_emb_cache_dir=None,
                 graph_type='static',
                 merge_strategy="tailhead", edge_strategy=None,
                 seed=None,
                 word_emb_size=300, share_vocab=True, lower_case=True,
                 thread_number=1, port=9000,
                 dynamic_graph_type=None,
                 dynamic_init_topology_builder=None,
                 dynamic_init_topology_aux_args=None,
                 for_inference=None,
                 reused_vocab_model=None):
        # Initialize the dataset. If the preprocessed files are not found, then do the preprocessing and save them.
        super(JobsDataset, self).__init__(root_dir=root_dir, topology_builder=topology_builder,
                                          topology_subdir=topology_subdir, graph_type=graph_type,
                                          edge_strategy=edge_strategy, merge_strategy=merge_strategy,
                                          share_vocab=share_vocab, lower_case=lower_case,
                                          pretrained_word_emb_name=pretrained_word_emb_name, pretrained_word_emb_url=pretrained_word_emb_url, pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
                                          seed=seed, word_emb_size=word_emb_size,
                                          thread_number=thread_number, port=port,
                                          dynamic_graph_type=dynamic_graph_type,
                                          dynamic_init_topology_builder=dynamic_init_topology_builder,
                                          dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
                                          for_inference=for_inference,
                                          reused_vocab_model=reused_vocab_model)

Customizing downloading
----------
Downloading can be decomposed into 2 steps: 1) check whether file exist and 2) download the missing files.
To customize checking, the file names must be specified:

.. code-block::

    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {'train': 'train.txt', 'test': 'test.txt'}


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
check for split ratio. It first checks if the processed files exist, and directly load these files if they exist in the file
system. Otherwise it will perform several pre-processing steps, namely ``build_topology``, ``build_vocab`` and ``vectorization``.

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


``build_topology`` builds text graph for each ``DataItem`` in the dataset and bind it to the corresponding ``DataItem``
object. This routine usually involves functions provided by the ``GraphConstruction`` module. Besides, since the
construction of each individual text graph is independent of each other, the construction of multiple graphs can be done
concurrently, which involves the multiprocessing module of Python.

``build_vocab`` takes all the tokens that have appeared in the data items and build a vocabulary out of it.
By default, the ``VocabModel`` in ``graph4nlp.utils.vocab_utils.VocabModel`` takes the responsibility of constructing
a vocabulary and represents the vocabulary itself. The constructed vocabulary will become a member of the ``Dataset``
instance.

``vectorization`` is a lookup step, which converts the tokens from ASCII characters to word embeddings. Since there are
various ways to assign embedding vectors to tokens, this step is usually overridden by the downstream classes.

In Jobs, these pre-processing steps are implemented in its base classes: ``Text2TextDataset`` and ``Dataset``:

.. code-block::

    class Dataset:
        def build_topology(self, data_items):
            """
            Build graph topology for each item in the dataset. The generated graph is bound to the `graph` attribute of the
            DataItem.
            """
            total = len(data_items)
            thread_number = min(total, self.thread_number)
            pool = Pool(thread_number)
            res_l = []
            for i in range(thread_number):
                start_index = total * i // thread_number
                end_index = total * (i + 1) // thread_number

                """
                data_items, topology_builder,
                                    graph_type, dynamic_graph_type, dynamic_init_topology_builder,
                                    merge_strategy, edge_strategy, dynamic_init_topology_aux_args,
                                    lower_case, tokenizer, port, timeout
                """
                r = pool.apply_async(self._build_topology_process,
                                     args=(data_items[start_index:end_index], self.topology_builder, self.graph_type,
                                           self.dynamic_graph_type, self.dynamic_init_topology_builder,
                                           self.merge_strategy, self.edge_strategy, self.dynamic_init_topology_aux_args,
                                           self.lower_case, self.tokenizer, self.port, self.timeout))
                res_l.append(r)
            pool.close()
            pool.join()

            data_items = []
            for i in range(thread_number):
                res = res_l[i].get()
                for data in res:
                    if data.graph is not None:
                        data_items.append(data)

            return data_items

        def build_vocab(self):
            """
            Build the vocabulary. If `self.use_val_for_vocab` is `True`, use both training set and validation set for building
            the vocabulary. Otherwise only the training set is used.

            """
            data_for_vocab = self.train
            if self.use_val_for_vocab:
                data_for_vocab = self.val + data_for_vocab

            vocab_model = VocabModel.build(saved_vocab_file=self.processed_file_paths['vocab'],
                                           data_set=data_for_vocab,
                                           tokenizer=self.tokenizer,
                                           lower_case=self.lower_case,
                                           max_word_vocab_size=self.max_word_vocab_size,
                                           min_word_vocab_freq=self.min_word_vocab_freq,
                                           share_vocab=self.share_vocab,
                                           pretrained_word_emb_name=self.pretrained_word_emb_name,
                                           pretrained_word_emb_url=self.pretrained_word_emb_url,
                                           pretrained_word_emb_cache_dir=self.pretrained_word_emb_cache_dir,
                                           target_pretrained_word_emb_name=self.target_pretrained_word_emb_name,
                                           target_pretrained_word_emb_url=self.target_pretrained_word_emb_url,
                                           word_emb_size=self.word_emb_size)
            self.vocab_model = vocab_model

            return self.vocab_model

    class Text2TextDataset:
        def vectorization(self, data_items):
            if self.topology_builder == IEBasedGraphConstruction:
                use_ie = True
            else:
                use_ie = False
            for item in data_items:
                graph: GraphData = item.graph
                token_matrix = []
                for node_idx in range(graph.get_node_num()):
                    node_token = graph.node_attributes[node_idx]['token']
                    node_token_id = self.vocab_model.in_word_vocab.getIndex(node_token, use_ie)
                    graph.node_attributes[node_idx]['token_id'] = node_token_id

                    token_matrix.append([node_token_id])
                if self.topology_builder == IEBasedGraphConstruction:
                    for i in range(len(token_matrix)):
                        token_matrix[i] = np.array(token_matrix[i][0])
                    token_matrix = pad_2d_vals_no_size(token_matrix)
                    token_matrix = torch.tensor(token_matrix, dtype=torch.long)
                    graph.node_features['token_id'] = token_matrix
                    pass
                else:
                    token_matrix = torch.tensor(token_matrix, dtype=torch.long)
                    graph.node_features['token_id'] = token_matrix

                if use_ie and 'token' in graph.edge_attributes[0].keys():
                    edge_token_matrix = []
                    for edge_idx in range(graph.get_edge_num()):
                        edge_token = graph.edge_attributes[edge_idx]['token']
                        edge_token_id = self.vocab_model.in_word_vocab.getIndex(edge_token, use_ie)
                        graph.edge_attributes[edge_idx]['token_id'] = edge_token_id
                        edge_token_matrix.append([edge_token_id])
                    if self.topology_builder == IEBasedGraphConstruction:
                        for i in range(len(edge_token_matrix)):
                            edge_token_matrix[i] = np.array(edge_token_matrix[i][0])
                        edge_token_matrix = pad_2d_vals_no_size(edge_token_matrix)
                        edge_token_matrix = torch.tensor(edge_token_matrix, dtype=torch.long)
                        graph.edge_features['token_id'] = edge_token_matrix

                tgt = item.output_text
                tgt_token_id = self.vocab_model.out_word_vocab.to_index_sequence(tgt)
                tgt_token_id.append(self.vocab_model.out_word_vocab.EOS)
                tgt_token_id = np.array(tgt_token_id)
                item.output_np = tgt_token_id

Customizing batching
-----------
The runtime iteration over dataset is performed by PyTorch's dataloader. And since the basic composing element is
``DataItem``, it is our job to convert the low-level list of ``DataItem`` fetched by ``torch.DataLoader`` to the batch
data we want.
``Dataset.collate_fn()`` is designed to do this job.

.. code-block::

    class Text2TextDataset:
        @staticmethod
        def collate_fn(data_list: [Text2TextDataItem]):
            graph_list = [item.graph for item in data_list]
            graph_data = to_batch(graph_list)

            output_numpy = [deepcopy(item.output_np) for item in data_list]
            output_str = [deepcopy(item.output_text.lower().strip()) for item in data_list]
            output_pad = pad_2d_vals_no_size(output_numpy)

            tgt_seq = torch.from_numpy(output_pad).long()
            return {
                "graph_data": graph_data,
                "tgt_seq": tgt_seq,
                "output_str": output_str
            }

It takes in a list of DataItem and returns the expected type of data required by the model. Interested readers may
refer to the examples we provided in the source code for practical usages.
