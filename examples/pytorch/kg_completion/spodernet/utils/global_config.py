import os
from collections import namedtuple
from spodernet.utils.logger import Logger
log = Logger('global_config.py.txt')

class Backends:
    TORCH = 'pytorch'
    TENSORFLOW = 'tensorflow'
    TEST = 'test'
    CNTK = 'cntk'


class Config:
    dropout = 0.0
    batch_size = 128
    learning_rate = 0.001
    backend = Backends.TORCH
    L2 = 0.000
    cuda = False
    embedding_dim = 128
    hidden_size = 256
    input_dropout = 0.0
    feature_map_dropout = 0.0
    use_conv_transpose = False
    use_bias = True
    optimizer = 'adam'
    learning_rate_decay = 1.0
    label_smoothing_epsilon = 0.1
    epochs = 1000
    dataset = None
    process = False
    model_name = None

    @staticmethod
    def parse_argv(argv):
        file_name = argv[0]
        args = argv[1:]
        assert len(args) % 2 == 0, 'Global parser expects an even number of arguments.'
        values = []
        names = []
        for i, token in enumerate(args):
            if i % 2 == 0:
                names.append(token)
            else:
                values.append(token)

        for i in range(len(names)):
            if names[i] in alias2params:
                log.debug('Replaced parameters alias {0} with name {1}', names[i], alias2params[names[i]])
                names[i] = alias2params[names[i]]

        for i in range(len(names)):
            name = names[i]
            if name[:2] == '--': continue
            if name not in params2type:
                log.info('List of possible parameters: {0}', params2type.keys())
                log.error('Parameter {0} does not exist. Prefix your custom parameters with -- to skip parsing for global config', name)
            values[i] = params2type[name](values[i])

        for name, value in zip(names, values):
            if name[:2] == '--': continue
            params2field[name](value)
            log.info('Set parameter {0} to {1}', name, value)

    input_dropout = 0.0
    feature_map_dropout = 0.0
    use_transposed_convolutions = False
    use_bias = True

params2type = {}
params2type['learning_rate'] = lambda x: float(x)
params2type['learning_rate_decay'] = lambda x: float(x)
params2type['dropout'] = lambda x: float(x)
params2type['batch_size'] = lambda x: int(x)
params2type['L2'] = lambda x: float(x)
params2type['embedding_dim'] = lambda x: int(x)
params2type['hidden_size'] = lambda x: int(x)
params2type['input_dropout'] = lambda x: float(x)
params2type['label_smoothing_epsilon'] = lambda x: float(x)
params2type['feature_map_dropout'] = lambda x: float(x)
params2type['use_conv_transpose'] = lambda x: x.lower() == 'true' or x == '1'
params2type['use_bias'] = lambda x: x.lower() == 'true' or x == '1'
params2type['optimizer'] = lambda x: x
params2type['epochs'] = lambda x: int(x)
params2type['dataset'] = lambda x: x
params2type['model_name'] = lambda x: x
params2type['process'] = lambda x: x.lower() == 'true' or x == '1'

alias2params = {}
alias2params['lr'] = 'learning_rate'
alias2params['lr_decay'] = 'learning_rate_decay'
alias2params['l2'] = 'L2'
alias2params['input_drop'] = 'input_dropout'
alias2params['hidden_drop'] = 'dropout'
alias2params['feat_drop'] = 'feature_map_dropout'
alias2params['bias'] = 'use_bias'
alias2params['conv_trans'] = 'use_conv_transpose'
alias2params['opt'] = 'optimizer'
alias2params['label_smoothing'] = 'label_smoothing_epsilon'
alias2params['model'] = 'model_name'



params2field = {}
params2field['learning_rate'] = lambda x: setattr(Config, 'learning_rate', x)
params2field['learning_rate_decay'] = lambda x: setattr(Config, 'learning_rate_decay', x)
params2field['dropout'] = lambda x: setattr(Config, 'dropout', x)
params2field['batch_size'] = lambda x: setattr(Config, 'batch_size', x)
params2field['L2'] = lambda x: setattr(Config, 'L2', x)
params2field['embedding_dim'] = lambda x: setattr(Config, 'embedding_dim', x)
params2field['hidden_size'] = lambda x: setattr(Config, 'hidden_size', x)
params2field['input_dropout'] = lambda x: setattr(Config, 'input_dropout', x)
params2field['feature_map_dropout'] = lambda x: setattr(Config, 'feature_map_dropout', x)
params2field['use_conv_transpose'] = lambda x: setattr(Config, 'use_conv_transpose', x)
params2field['use_bias'] = lambda x: setattr(Config, 'use_bias', x)
params2field['optimizer'] = lambda x: setattr(Config, 'optimizer', x)
params2field['label_smoothing_epsilon'] = lambda x: setattr(Config, 'label_smoothing_epsilon', x)
params2field['epochs'] = lambda x: setattr(Config, 'epochs', x)
params2field['dataset'] = lambda x: setattr(Config, 'dataset', x)
params2field['process'] = lambda x: setattr(Config, 'process', x)
params2field['model_name'] = lambda x: setattr(Config, 'model_name', x)


