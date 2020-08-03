import os
import json
import sys
from . import constants as Constants


class DummyLogger(object):
    def __init__(self, config, dirname=None, pretrained=None):
        self.config = config
        if dirname is None:
            if pretrained is None:
                raise Exception('Either --dir or --pretrained needs to be specified.')
            self.dirname = pretrained
        else:
            self.dirname = dirname
            if os.path.exists(dirname):
                # raise Exception('Directory already exists: {}'.format(dirname))
                print('Directory already exists: {}'.format(dirname))
            else:
                os.makedirs(dirname)
                os.mkdir(os.path.join(dirname, 'metrics'))
            self.log_json(config, os.path.join(self.dirname, Constants._CONFIG_FILE))
        if config['logging']:
            self.f_metric = open(os.path.join(self.dirname, 'metrics', 'metrics.log'), 'a')

    def log_json(self, data, filename, mode='w'):
        with open(filename, mode) as outfile:
            outfile.write(json.dumps(data, indent=4, ensure_ascii=False))

    def log(self, data, filename):
        print(data)

    def write_to_file(self, text):
        if self.config['logging']:
            self.f_metric.writelines(text + '\n')
            self.f_metric.flush()

    def close(self):
        if self.config['logging']:
            self.f_metric.close()

class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
