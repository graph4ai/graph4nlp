from enum import IntEnum
from os.path import join

import os
import datetime
import numpy as np
import time

# util functions start
#
# these function also exist in util.py,
# but since logger is imported everywere these function need to be included here

def get_home_path():
    return os.environ['HOME']

def get_logger_path():
    return join(get_home_path(), '.data', 'log_files')

def make_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# util functions end
class GlobalLogger:
    timestr = None
    global_logger_path = None
    f_global_logger = None

    @staticmethod
    def init():
        GlobalLogger.timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(join(get_logger_path(), 'full_logs')):
            os.mkdir(join(get_logger_path(), 'full_logs'))
        GlobalLogger.global_logger_path = join(get_logger_path(), 'full_logs', GlobalLogger.timestr +  '.txt')
        GlobalLogger.f_global_logger = open(GlobalLogger.global_logger_path, 'w')

    @staticmethod
    def flush():
        GlobalLogger.f_global_logger.close()
        GlobalLogger.f_global_logger = open(GlobalLogger.global_logger_path, 'a')

    def __del__(self):
        GlobalLogger.f_global_logger.close()

class LogLevel(IntEnum):
    STATISTICAL = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4

class Logger:
    GLOBAL_LOG_LEVEL = LogLevel.INFO
    LOG_PROPABILITY = 0.05
    USE_GLOBAL_STATISTICAL_LOG_PROBABILITY = False
    PRINT_COUNT = 2

    def __init__(self, file_name, write_type='w'):
        path = join(get_logger_path(), file_name)
        path_statistical = join(get_logger_path(), 'statistical_' + file_name)
        self.path = path
        make_dirs_if_not_exists(get_logger_path())
        self.f = open(path, write_type)
        self.f_statistical = open(path_statistical, write_type)
        self.rdm = np.random.RandomState(234234)
        self.debug('Created log file at: {0} with write type: {1}'.format(path, write_type))
        self.once_dict = {}

    def __del__(self):
        self.f.close()
        self.f_statistical.close()

    def wrap_message(self, message, log_level, *args):
        return '{0} ({2}): {1}'.format(datetime.datetime.now(), message.format(*args), log_level.name)

    def statistical(self, message, p, *args):
        if Logger.GLOBAL_LOG_LEVEL == LogLevel.STATISTICAL:
            self._log_statistical(message, p, *args)

    def debug(self, message, *args):
        self._log(message, LogLevel.DEBUG, *args)

    def info_once(self, message, *args):
        if LogLevel.INFO < Logger.GLOBAL_LOG_LEVEL: return
        if message not in self.once_dict: self.once_dict[message] = 0
        if self.once_dict[message] < Logger.PRINT_COUNT:
            self.once_dict[message] += 1
            self._log(message, LogLevel.INFO, *args)

    def debug_once(self, message, *args):
        if LogLevel.DEBUG < Logger.GLOBAL_LOG_LEVEL: return
        if message not in self.once_dict: self.once_dict[message] = 0
        if self.once_dict[message] < Logger.PRINT_COUNT:
            self.once_dict[message] += 1
            self._log(message, LogLevel.DEBUG, *args)

    def info(self, message, *args):
        self._log(message, LogLevel.INFO, *args)

    def warning(self, message, *args):
        self._log(message, LogLevel.WARNING, *args)

    def error(self, message, *args):
        self._log(message, LogLevel.ERROR, *args)
        raise Exception(message.format(*args))

    def _log_statistical(self, message, p, *args):
        rdm_num = self.rdm.rand()
        if Logger.USE_GLOBAL_STATISTICAL_LOG_PROBABILITY:
            if rdm_num < Logger.LOG_PROPABILITY:
                message = self.wrap_message(message, LogLevel.STATISTICAL, *args)
                self.f_statistical.write(message + '\n')
        else:
            if rdm_num < p:
                message = self.wrap_message(message, LogLevel.STATISTICAL, *args)
                self.f_statistical.write(message + '\n')

    def _log(self, message, log_level=LogLevel.INFO, *args):
        if log_level >= Logger.GLOBAL_LOG_LEVEL:
            message = self.wrap_message(message, log_level, *args)
            if message.strip() != '':
                print(message)
                self.f.write(message + '\n')
                if GlobalLogger.f_global_logger is None: GlobalLogger.init()
                GlobalLogger.f_global_logger.write(message + '\n')


