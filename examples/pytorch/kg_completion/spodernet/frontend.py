from itertools import chain

from spodernet.utils.global_config import Config, Backends

from spodernet.utils.logger import Logger
log = Logger('frontend.py.txt')


class Model(object):

    def __init__(self, input_module=None):
        self.modules = []
        self.input_module = input_module
        self.module = self

    def add(self, module):
        self.modules.append(module)

    def forward(self, str2var, *inputs):
        outputs = inputs
        if inputs == None:
            outputs = []
        for module in self.modules:
            outputs = module.forward(str2var, *outputs)
        return outputs

class Trainer(object):
    def __init__(self, model):
        self.model = model

        self.trainer_backend = None
        self.train_func = None
        self.eval_func = None
        if Config.backend == Backends.TENSORFLOW:
            from spodernet.backends.tfbackend import TFTrainer
            self.trainer_backend = TFTrainer(model)
            self.train_func = lambda _, batch, epochs, iterations: self.trainer_backend.train_model(batch, epochs, iterations)
            self.eval_func = lambda _, batch, iterations: self.trainer_backend.eval_model(batch, iterations)
        elif Config.backend == Backends.TORCH:
            from spodernet.backends.torchbackend import train_model, eval_model
            self.train_func = train_model
            self.eval_func = eval_model

    def train(self, batcher, epochs=1, iterations=None):
        self.train_func(self.model, batcher, epochs, iterations)

    def evaluate(self, batcher, iterations=None):
        self.eval_func(self.model, batcher, iterations)

class AbstractModel(object):

    def __init__(self):
        super(AbstractModel, self).__init__()
        self.input_str_args = None
        self.output_str_args = None
        self.used_keys = None

    def forward(self, str2var, *args):
        raise NotImplementedError("Classes that inherit from AbstractModel need to implement the forward method.")

    @property
    def modules(self):
        raise NotImplementedError("Classes that inherit from AbstractModel need to overrite the modules property.")

    def expected_str2var_keys(self, str2var, keys):
        self.used_keys = keys
        for key in keys:
            if key not in str2var:
                log.error('Variable with name {0} expected, but not found in str2variable dict with keys {1}'.format(key, str2var.keys()))

    def expected_str2var_keys_oneof(self, str2var, keys):
        self.used_keys = keys
        one_exists = False
        for key in keys:
            if key in str2var:
                one_exists = True
        if not one_exists:
            log.error('At least one of these variable was expected: {0}. But str2var only has these variables: {1}.', keys, str2var.keys())

    def expected_args(self, str_arg_names, str_arg_description):
        log.debug_once('Expected args {0}'.format(str_arg_names))
        log.debug_once('Info for the expected arguments: {0}'.format(str_arg_description))
        self.input_str_args = str_arg_names

    def generated_outputs(self, str_output_names, str_output_description):
        log.debug_once('Generated outputs: {0}'.format(str_output_names))
        log.debug_once('Info for the provided outputs: {0}'.format(str_output_description))
        self.output_str_args = str_output_names
        self.used_keys
        self.input_str_args
        self.output_str_args
        message = '{0} + {1} -> {2}'.format(self.used_keys, self.input_str_args, self.output_str_args)
        log.info_once(message)


class Embedding(object):
    def __init__(self, embedding_size, num_embeddings, scope=None):
        self.embedding_size = embedding_size
        self.scope = scope
        self.num_embeddings = num_embeddings

        self.module = None
        if Config.backend == Backends.TENSORFLOW:
            from spodernet.backends.tfmodels import TFEmbedding
            self.module = TFEmbedding(embedding_size, num_embeddings, scope)
        elif Config.backend == Backends.TORCH:
            from spodernet.backends.torchmodels import TorchEmbedding
            self.module = TorchEmbedding(embedding_size, num_embeddings)
            self.modules = [self.module]

    def forward(self, str2var, *args):
        return self.module.forward(str2var, *args)


class PairedBiDirectionalLSTM(object):

    def __init__(self, input_size, hidden_size, scope=None, conditional_encoding=True):
        super(PairedBiDirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.scope = scope

        self.module = None
        if Config.backend == Backends.TENSORFLOW:
            from spodernet.backends.tfmodels import TFPairedBiDirectionalLSTM
            self.module = TFPairedBiDirectionalLSTM(hidden_size, scope, conditional_encoding)
        elif Config.backend == Backends.TORCH:
            from spodernet.backends.torchmodels import TorchPairedBiDirectionalLSTM, TorchVariableLengthOutputSelection
            model = Model()
            model.add(TorchPairedBiDirectionalLSTM(input_size, hidden_size, conditional_encoding=conditional_encoding))
            model.add(TorchVariableLengthOutputSelection())

            self.module = model
            self.modules = model.modules

    def forward(self, str2var, *args):
        return self.module.forward(str2var, *args)


class SoftmaxCrossEntropy(object):
    def __init__(self, input_size, num_labels):
        super(SoftmaxCrossEntropy, self).__init__()
        self.num_labels = num_labels

        self.module = None
        if Config.backend == Backends.TENSORFLOW:
            from spodernet.backends.tfmodels import TFSoftmaxCrossEntropy
            self.module = TFSoftmaxCrossEntropy(num_labels)
        elif Config.backend == Backends.TORCH:
            from spodernet.backends.torchmodels import TorchSoftmaxCrossEntropy
            self.module = TorchSoftmaxCrossEntropy(input_size, num_labels)
            self.modules  = [self.module]

    def forward(self, str2var, *args):
        return self.module.forward(str2var, *args)
