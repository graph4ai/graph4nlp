Semantic Parsing Tutorial
===================================


Introduction
------------


In this tutorial demo, we will use the Graph4NLP library to build a GNN-based semantic parsing model. The model consists of

- graph construction module (e.g., node embedding based dynamic graph)
- graph embedding module (e.g., Bi-Sep GAT)
- predictoin module (e.g., RNN decoder with attention, copy and coverage mechanisms)

We will use the built-in Graph2Seq model APIs to build the model, and evaluate it on the Jobs dataset.
The full example can be downloaded from `Semantic parsing notebook <https://github.com/graph4ai/graph4nlp_demo/blob/main/SIGIR2021_demo/semantic_parsing.ipynb>`__.



Environment setup
------

Please follow the instructions `here <https://github.com/graph4ai/graph4nlp_demo#environment-setup>`__ to set up the environment.




Build the model handler
----


Let's build a model handler which will do a bunch of things including setting up dataloader, model, optimizer, evaluation metrics, train/val/test loops, and so on.

We will call the Graph2Seq model API which implements a GNN-based encoder and LSTM-based decoder.

When setting up the dataloader, users will need to call the dataset API which will preprocess the data, e.g., calling the graph construction module, building the vocabulary, tensorizing the data. Users will need to specify the graph construction type when calling the dataset API.

Users can build their customized dataset APIs by inheriting our low-level dataset APIs. We provide low-level dataset APIs to support various scenarios (e.g., `Text2Label`, `Sequence2Labeling`, `Text2Text`, `Text2Tree`, `DoubleText2Text`).



.. code-block:: python

    class Jobs:
    def __init__(self, opt):
        super(Jobs, self).__init__()
        self.opt = opt
        self.use_copy = self.opt["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_coverage = self.opt["decoder_args"]["rnn_decoder_share"]["use_coverage"]
        self._build_device(self.opt)
        self._build_dataloader()
        self._build_model()
        self._build_loss_function()
        self._build_optimizer()
        self._build_evaluation()

    def _build_device(self, opt):
        seed = opt["seed"]
        np.random.seed(seed)
        if opt["use_gpu"] != 0 and torch.cuda.is_available():
            print('[ Using CUDA ]')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn
            cudnn.benchmark = True
            device = torch.device('cuda' if opt["gpu"] < 0 else 'cuda:%d' % opt["gpu"])
        else:
            print('[ Using CPU ]')
            device = torch.device('cpu')
        self.device = device

    def _build_dataloader(self):
        if self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "dependency":
            topology_builder = DependencyBasedGraphConstruction
            graph_type = 'static'
            dynamic_init_topology_builder = None
        elif self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "constituency":
            topology_builder = ConstituencyBasedGraphConstruction
            graph_type = 'static'
            dynamic_init_topology_builder = None
        elif self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "node_emb":
            topology_builder = NodeEmbeddingBasedGraphConstruction
            graph_type = 'dynamic'
            dynamic_init_topology_builder = None
        elif self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"] == "node_emb_refined":
            topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
            graph_type = 'dynamic'
            dynamic_init_graph_type = self.opt["graph_construction_args"]["graph_construction_private"][
                "dynamic_init_graph_type"]
            if dynamic_init_graph_type is None or dynamic_init_graph_type == 'line':
                dynamic_init_topology_builder = None
            elif dynamic_init_graph_type == 'dependency':
                dynamic_init_topology_builder = DependencyBasedGraphConstruction
            elif dynamic_init_graph_type == 'constituency':
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            else:
                raise RuntimeError('Define your own dynamic_init_topology_builder')
        else:
            raise NotImplementedError("Define your topology builder.")


        # Call the TREC dataset API
        dataset = JobsDataset(root_dir=self.opt["graph_construction_args"]["graph_construction_share"]["root_dir"],
                              pretrained_word_emb_name=self.opt["pretrained_word_emb_name"],
                              pretrained_word_emb_cache_dir=self.opt["pretrained_word_emb_cache_dir"],
                              merge_strategy=self.opt["graph_construction_args"]["graph_construction_private"]["merge_strategy"],
                              edge_strategy=self.opt["graph_construction_args"]["graph_construction_private"]["edge_strategy"],
                              seed=self.opt["seed"], word_emb_size=self.opt["word_emb_size"],
                              share_vocab=self.opt["graph_construction_args"]["graph_construction_share"]["share_vocab"],
                              graph_type=graph_type, topology_builder=topology_builder,
                              topology_subdir=self.opt["graph_construction_args"]["graph_construction_share"]["topology_subdir"],
                              thread_number=self.opt["graph_construction_args"]["graph_construction_share"]["thread_number"],
                              dynamic_graph_type=self.opt["graph_construction_args"]["graph_construction_share"]["graph_type"],
                              dynamic_init_topology_builder=dynamic_init_topology_builder, dynamic_init_topology_aux_args=None)

        self.train_dataloader = DataLoader(dataset.train, batch_size=self.opt["batch_size"], shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.test, batch_size=self.opt["batch_size"], shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
        self.vocab = dataset.vocab_model

    def _build_model(self):
        # Call the Graph2Seq model API
        self.model = Graph2Seq.from_args(self.opt, self.vocab).to(self.device)

    def _build_loss_function(self):
        # Call the Graph2Seq loss API
        self.loss = Graph2SeqLoss(ignore_index=self.vocab.out_word_vocab.PAD, use_coverage=self.use_coverage, coverage_weight=0.3)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.opt["learning_rate"])

    def _build_evaluation(self):
        self.metrics = [ExpressionAccuracy()]

    def train(self):
        max_score = -1
        self._best_epoch = -1
        for epoch in range(self.opt["epochs"]):
            self.model.train()
            self.train_epoch(epoch, split="train")
            self._adjust_lr(epoch)
            if epoch >= 0:
                score = self.evaluate(split="test")
                if score >= max_score:
                    print("Best model saved, epoch {}".format(epoch))
                    self.save_checkpoint("best.pth")
                    self._best_epoch = epoch
                max_score = max(max_score, score)
            if self._stop_condition(epoch):
                break
        return max_score

    def _stop_condition(self, epoch, patience=20):
        return epoch > patience + self._best_epoch

    def _adjust_lr(self, epoch):
        def set_lr(optimizer, decay_factor):
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] * decay_factor

        epoch_diff = epoch - self.opt["lr_start_decay_epoch"]
        if epoch_diff >= 0 and epoch_diff % self.opt["lr_decay_per_epoch"] == 0:
            if self.opt["learning_rate"] > self.opt["min_lr"]:
                set_lr(self.optimizer, self.opt["lr_decay_rate"])
                self.opt["learning_rate"] = self.opt["learning_rate"] * self.opt["lr_decay_rate"]
                print("Learning rate adjusted: {:.5f}".format(self.opt["learning_rate"]))

    def train_epoch(self, epoch, split="train"):
        assert split in ["train"]
        print("Start training in split {}, Epoch: {}".format(split, epoch))
        loss_collect = []
        dataloader = self.train_dataloader
        step_all_train = len(dataloader)
        for step, data in enumerate(dataloader):
            graph, tgt, gt_str = data["graph_data"], data["tgt_seq"], data["output_str"]
            graph = graph.to(self.device)
            tgt = tgt.to(self.device)
            oov_dict = None
            if self.use_copy:
                oov_dict, tgt = prepare_ext_vocab(graph, self.vocab, gt_str=gt_str, device=self.device)

            prob, enc_attn_weights, coverage_vectors = self.model(graph, tgt, oov_dict=oov_dict)
            loss = self.loss(logits=prob, label=tgt, enc_attn_weights=enc_attn_weights, coverage_vectors=coverage_vectors)
            loss_collect.append(loss.item())
            if step % self.opt["loss_display_step"] == 0 and step != 0:
                print("Epoch {}: [{} / {}] loss: {:.3f}".format(epoch, step, step_all_train, np.mean(loss_collect)))
                loss_collect = []
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, split="val"):
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["val", "test"]
        dataloader = self.val_dataloader if split == "val" else self.test_dataloader
        for data in dataloader:
            graph, tgt, gt_str = data["graph_data"], data["tgt_seq"], data["output_str"]
            graph = graph.to(self.device)
            if self.use_copy:
                oov_dict = prepare_ext_vocab(batch_graph=graph, vocab=self.vocab, device=self.device)
                ref_dict = oov_dict
            else:
                oov_dict = None
                ref_dict = self.vocab.out_word_vocab

            prob, _, _ = self.model(graph, oov_dict=oov_dict)
            pred = prob.argmax(dim=-1)

            pred_str = wordid2str(pred.detach().cpu(), ref_dict)
            pred_collect.extend(pred_str)
            gt_collect.extend(gt_str)

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        print("Evaluation accuracy in `{}` split: {:.3f}".format(split, score))
        return score

    @torch.no_grad()
    def translate(self):
        self.model.eval()

        pred_collect = []
        gt_collect = []
        dataloader = self.test_dataloader
        for data in dataloader:
            graph, tgt, gt_str = data["graph_data"], data["tgt_seq"], data["output_str"]
            graph = graph.to(self.device)
            if self.use_copy:
                oov_dict = prepare_ext_vocab(batch_graph=graph, vocab=self.vocab, device=self.device)
                ref_dict = oov_dict
            else:
                oov_dict = None
                ref_dict = self.vocab.out_word_vocab

            pred = self.model.translate(batch_graph=graph, oov_dict=oov_dict, beam_size=4, topk=1)

            pred_ids = pred[:, 0, :]  # we just use the top-1

            pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)

            pred_collect.extend(pred_str)
            gt_collect.extend(gt_str)

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        return score

    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.opt["checkpoint_save_path"], checkpoint_name)
        self.model.load_state_dict(torch.load(checkpoint_path))

    def save_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.opt["checkpoint_save_path"], checkpoint_name)
        if not os.path.exists(self.opt["checkpoint_save_path"]):
            os.makedirs(self.opt["checkpoint_save_path"], exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)


Run the model
----

.. code-block:: python

    runner = Jobs(opt)
    max_score = runner.train()
    runner.load_checkpoint("best.pth")
    test_score = runner.translate()


.. parsed-literal::

    [ Using CPU ]
    Start training in split train, Epoch: 0
    Epoch 0: [10 / 21] loss: 3.938
    Epoch 0: [20 / 21] loss: 2.506
    Evaluation accuracy in `test` split: 0.000
    Best model saved, epoch 0
    Start training in split train, Epoch: 1
    Epoch 1: [10 / 21] loss: 1.845
    Epoch 1: [20 / 21] loss: 1.487
    Evaluation accuracy in `test` split: 0.000
    Best model saved, epoch 1
    Start training in split train, Epoch: 2
    Epoch 2: [10 / 21] loss: 1.198
    Epoch 2: [20 / 21] loss: 1.104
    Evaluation accuracy in `test` split: 0.100
    Best model saved, epoch 2
    ......
