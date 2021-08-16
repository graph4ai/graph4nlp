Knowledge Graph Completion Tutorial
===================================

Introduction
------------

In this tutorial demo, we will use the Graph4NLP library to build a
GNN-based knowledge graph completion model. The model consists of

-  graph embedding module (e.g., GGNN)
-  predictoin module (e.g., DistMult decoder)

We will use the built-in Graph2Seq model APIs to build the model, and
evaluate it on the Kinship dataset. The full example can be downloaded from
`knowledge graph completion notebook <https://github.com/graph4ai/graph4nlp_demo/tree/main/KDD2021_demo/kg_completion>`__


Environment setup
-----------------

Create virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  conda create –name g4l python=3.7
-  conda activate g4l

Install graph4nlp library via pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure that at least PyTorch (>=1.6.0) is installed:

.. code:: bash

   $ python -c "import torch; print(torch.__version__)"
   >>> 1.6.0

Find the CUDA version PyTorch was installed with (for GPU users):

.. code:: bash

   $ python -c "import torch; print(torch.version.cuda)"
   >>> 10.2

Install the relevant dependencies:

``torchtext`` is needed since Graph4NLP relies on it to implement
embeddings. Please pay attention to the PyTorch requirements before
installing ``torchtext`` with the following script! For detailed version
matching please refer `here <https://pypi.org/project/torchtext/>`__.

.. code:: bash

   pip install torchtext # >=0.7.0

Install Graph4NLP

.. code:: bash

   pip install graph4nlp${CUDA}

where ``${CUDA}`` should be replaced by the specific CUDA version
(``none`` (CPU version), ``"-cu92"``, ``"-cu101"``, ``"-cu102"``,
``"-cu110"``). The following table shows the concrete command lines. For
CUDA 11.1 users, please refer to ``Installation via source code``.

========= ===============================
Platform  Command
========= ===============================
CPU       ``pip install graph4nlp``
CUDA 9.2  ``pip install graph4nlp-cu92``
CUDA 10.1 ``pip install graph4nlp-cu101``
CUDA 10.2 ``pip install graph4nlp-cu102``
CUDA 11.0 ``pip install graph4nlp-cu110``
========= ===============================

Installation for KGC
~~~~~~~~~~~~~~~~~~~~


-  Download the default English model used by **spaCy**, which is
   installed in the previous step

.. code:: bash

   pip install spacy
   python -m spacy download en_core_web_sm
   pip install h5py
   pip install future

-  Run the preprocessing script for WN18RR and Kinship:
   ``sh kg_completion/preprocess.sh``

-  You can now run the model

Import packages
---------------

.. code:: python

    import torch
    import numpy as np
    import torch.backends.cudnn as cudnn
    
    from evaluation import ranking_and_hits
    from model import ConvE, Distmult, Complex, GGNNDistMult, GCNDistMult, GCNComplex
    
    from spodernet.preprocessing.pipeline import DatasetStreamer
    from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
    from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
    from spodernet.utils.global_config import Config, Backends
    from spodernet.utils.logger import Logger, LogLevel
    from spodernet.preprocessing.batching import StreamBatcher
    from spodernet.preprocessing.pipeline import Pipeline
    from spodernet.preprocessing.processors import TargetIdx2MultiTarget
    from spodernet.hooks import LossHook, ETAHook
    from spodernet.utils.util import Timer
    from spodernet.preprocessing.processors import TargetIdx2MultiTarget
    import argparse
    
    np.set_printoptions(precision=3)
    cudnn.benchmark = True

Data Preprocessing
------------------

This part we follow the implementaion of `ConvE <https://github.com/TimDettmers/ConvE>`_ to ensure the speed of processing data.

.. code:: python

    ''' Preprocess knowledge graph using spodernet. '''
    def preprocess(dataset_name, delete_data=False):
        full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
        train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
        dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
        test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)
    
        keys2keys = {}
        keys2keys['e1'] = 'e1' # entities
        keys2keys['rel'] = 'rel' # relations
        keys2keys['rel_eval'] = 'rel' # relations
        keys2keys['e2'] = 'e1' # entities
        keys2keys['e2_multi1'] = 'e1' # entity
        keys2keys['e2_multi2'] = 'e1' # entity
        input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
        d = DatasetStreamer(input_keys)
        d.add_stream_processor(JsonLoaderProcessors())
        d.add_stream_processor(DictKey2ListMapper(input_keys))
    
        # process full vocabulary and save it to disk
        d.set_path(full_path)
        p = Pipeline(args.data, delete_data, keys=input_keys, skip_transformation=True)
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_token_processor(AddToVocab())
        p.execute(d)
        p.save_vocabs()
    
    
        # process train, dev and test sets and save them to hdf5
        p.skip_transformation = False
        for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
            d.set_path(path)
            p.clear_processors()
            p.add_sent_processor(ToLower())
            p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
            p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
            p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
            p.execute(d)

Build Model
-----------

.. code:: python

    def main(args, model_path):
        if args.preprocess:
            preprocess(args.data, delete_data=True)
        input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
        p = Pipeline(args.data, keys=input_keys)
        p.load_vocabs()
        vocab = p.state['vocab']
    
        train_batcher = StreamBatcher(args.data, 'train', args.batch_size, randomize=True, keys=input_keys, loader_threads=args.loader_threads)
        dev_rank_batcher = StreamBatcher(args.data, 'dev_ranking', args.test_batch_size, randomize=False, loader_threads=args.loader_threads, keys=input_keys)
        test_rank_batcher = StreamBatcher(args.data, 'test_ranking', args.test_batch_size, randomize=False, loader_threads=args.loader_threads, keys=input_keys)
    
    
        data = []
        rows = []
        columns = []
        num_entities = vocab['e1'].num_token
        num_relations = vocab['rel'].num_token
    
        if args.preprocess:
            for i, str2var in enumerate(train_batcher):
                print("batch number:", i)
                for j in range(str2var['e1'].shape[0]):
                    for k in range(str2var['e2_multi1'][j].shape[0]):
                        if str2var['e2_multi1'][j][k] != 0:
                            data.append(str2var['rel'][j].cpu().tolist()[0])
                            rows.append(str2var['e1'][j].cpu().tolist()[0])
                            columns.append(str2var['e2_multi1'][j][k].cpu().tolist())
                        else:
                            break
    
            from graph4nlp.pytorch.data.data import GraphData, to_batch
            KG_graph = GraphData()
            KG_graph.add_nodes(num_entities)
            for e1, rel, e2 in zip(rows, data, columns):
                KG_graph.add_edge(e1, e2)
                eid = KG_graph.edge_ids(e1, e2)[0]
                KG_graph.edge_attributes[eid]['token'] = rel
    
            torch.save(KG_graph, '{}/processed/KG_graph.pt'.format(args.data))
            return
    
    
        if args.model is None:
            model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
        elif args.model == 'conve':
            model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
        elif args.model == 'distmult':
            model = Distmult(args, vocab['e1'].num_token, vocab['rel'].num_token)
        elif args.model == 'complex':
            model = Complex(args, vocab['e1'].num_token, vocab['rel'].num_token)
        elif args.model == 'ggnn_distmult':
            model = GGNNDistMult(args, vocab['e1'].num_token, vocab['rel'].num_token)
        elif args.model == 'gcn_distmult':
            model = GCNDistMult(args, vocab['e1'].num_token, vocab['rel'].num_token)
        elif args.model == 'gcn_complex':
            model = GCNComplex(args, vocab['e1'].num_token, vocab['rel'].num_token)
        else:
            raise Exception("Unknown model!")
    
        if args.model in ['ggnn_distmult', 'gcn_distmult', 'gcn_complex']:
            graph_path = '{}/processed/KG_graph.pt'.format(args.data)
            KG_graph = torch.load(graph_path)
            if Config.cuda:
                KG_graph = KG_graph.to('cuda')
        else:
            KG_graph = None
    
        train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))
    
        eta = ETAHook('train', print_every_x_batches=args.log_interval)
        train_batcher.subscribe_to_events(eta)
        train_batcher.subscribe_to_start_of_epoch_event(eta)
        train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=args.log_interval))
        if Config.cuda:
            model.cuda()
        if args.resume:
            model_params = torch.load(model_path)
            print(model)
            total_param_size = []
            params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
            for key, size, count in params:
                total_param_size.append(count)
                print(key, size, count)
            print(np.array(total_param_size).sum())
            model.load_state_dict(model_params)
            model.eval()
            ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation', kg_graph=KG_graph)
            ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation', kg_graph=KG_graph)
        else:
            model.init()
    
        total_param_size = []
        params = [value.numel() for value in model.parameters()]
        print(params)
        print(np.sum(params))
    
        best_mrr = 0
    
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        for epoch in range(args.epochs):
            model.train()
            for i, str2var in enumerate(train_batcher):
                opt.zero_grad()
                e1 = str2var['e1']
                rel = str2var['rel']
                e2_multi = str2var['e2_multi1_binary'].float()
                # label smoothing
                e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.size(1))
    
                pred = model.forward(e1, rel, KG_graph)
                loss = model.loss(pred, e2_multi)
                loss.backward()
                opt.step()
    
                train_batcher.state.loss = loss.cpu()
    
            model.eval()
            with torch.no_grad():
                if epoch % 2 == 0 and epoch > 0:
                    dev_mrr = ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation', kg_graph=KG_graph)
                    if dev_mrr > best_mrr:
                        best_mrr = dev_mrr
                        print('saving best model to {0}'.format(model_path))
                        torch.save(model.state_dict(), model_path)
                if epoch % 2 == 0:
                    if epoch > 0:
                        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation', kg_graph=KG_graph)

Config Setup
------------

.. code:: python

    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing/validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='kinship', help='Dataset to use: {FB15k-237, YAGO3-10, WN18RR, umls, nations, kinship}, default: FB15k-237')
    parser.add_argument('--l2', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--model', type=str, default='ggnn_distmult', help='Choose from: {conve, distmult, complex}')
    parser.add_argument('--direction_option', type=str, default='undirected', help='Choose from: {undirected, bi_sep, bi_fuse}')
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-drop', type=float, default=0.25, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr-decay', type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--loader-threads', type=int, default=4, help='How many loader threads to use for the batch loaders. Default: 4')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the dataset. Needs to be executed only once. Default: 4')
    parser.add_argument('--resume', action='store_true', help='Resume a model.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--hidden-size', type=int, default=9728, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
    
    parser.add_argument('--channels', type=int, default=200, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
    parser.add_argument('--kernel_size', type=int, default=5, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')


If you run the task for the first time, run with:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    args = parser.parse_args(args=['--data', 'kinship', '--model', 'ggnn_distmult', '--preprocess'])

    # parse console parameters and set global variables
    Config.backend = 'pytorch'
    Config.cuda = False
    Config.embedding_dim = args.embedding_dim
    
    model_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    
    torch.manual_seed(args.seed)

    main(args, model_path)


After preprocess the kinship data, then run:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    args = parser.parse_args(args=['--data', 'kinship', '--model', 'ggnn_distmult'])
    main(args, model_path)


Results on kinship
------------------

.. list-table:: BCELoss+GGNNDistmult
   :widths: 25 25 25 25
   :header-rows: 1

   * - Metrics
     - uni
     - bi_fuse
     - bi_sep
   * - Hits @1
     - 40.4
     - 39.4
     - 38.2
   * - Hits @10
     - 88.3
     - 88.8
     - 88.9
   * - MRR
     - 54.9
     - 54.8
     - 53.4
