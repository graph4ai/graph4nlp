import json
import os

from . import constants as Constants


class Logger:
    def __init__(self, dirname, config=None, overwrite=False, logging=True):
        self.logging = logging
        if os.path.exists(dirname):
            if not overwrite:
                raise Exception("Directory already exists: {}".format(dirname))
        else:
            os.makedirs(dirname)

        if config is not None:
            self.log_json(config, os.path.join(dirname, Constants._CONFIG_FILE))

        if logging:
            self.fout = open(os.path.join(dirname, Constants._SAVED_METRICS_FILE), "a")

    def log_json(self, data, filename, mode="w"):
        with open(filename, mode) as outfile:
            outfile.write(json.dumps(data, indent=4, ensure_ascii=False))

    def write(self, text):
        if self.logging:
            self.fout.writelines(text + "\n")
            self.fout.flush()

    def close(self):
        if self.logging:
            self.fout.close()
