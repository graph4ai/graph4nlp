from future import standard_library
standard_library.install_aliases()

from os.path import join, exists
import threading
from collections import namedtuple

import time
import datetime
import numpy as np
import queue
import pickle

from spodernet.utils.util import get_data_path, load_data, Timer
from spodernet.utils.global_config import Config, Backends
from spodernet.hooks import ETAHook
from spodernet.interfaces import IAtIterEndObservable, IAtEpochEndObservable, IAtEpochStartObservable, IAtBatchPreparedObservable
from spodernet.preprocessing.processors import DictConverter

from spodernet.utils.logger import Logger
log = Logger('batching.py.txt')

benchmark = False


class BatcherState(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.loss = None
        self.argmax = None
        self.pred = None
        self.batch_size = None
        self.current_idx = None
        self.current_epoch = None
        self.targets = None
        self.num_batches = None
        self.timer = None
        self.multi_labels = None


class DataLoaderSlave(threading.Thread):
    def __init__(self, stream_batcher, batchidx2paths, batchidx2start_end, randomize=False, paths=None, shard2batchidx=None, seed=None, shard_fractions=None, cache_size_GB=4):
        super(DataLoaderSlave, self).__init__()
        if randomize:
            assert seed is not None, 'For randomized data loading a seed needs to be set!'
        self.cache_size_GB = cache_size_GB
        self.stream_batcher = stream_batcher
        self.batchidx2paths = batchidx2paths
        self.batchidx2start_end = batchidx2start_end
        self.current_data = {}
        self.randomize = randomize
        self.num_batches = len(list(batchidx2paths.keys()))
        self.rdm = np.random.RandomState(234+seed)
        self.shard_fractions = shard_fractions
        self.shard2batchidx = shard2batchidx
        self.paths = paths
        self._stop = threading.Event()
        self.daemon = True
        self.t = Timer()
        self.batches_processes = 0
        self.cache_order = []

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def load_files_if_needed(self, current_paths):
        if isinstance(current_paths[0], list):
            for paths in current_paths:
                shuffle_idx = None
                for path in paths:
                    if path not in self.current_data:
                        ordered_data = load_data(path)
                        self.cache_order.append(path)
                        if shuffle_idx == None and self.randomize:
                            shuffle_idx = np.arange(ordered_data.shape[0])
                            self.rdm.shuffle(shuffle_idx)

                        if self.randomize:
                            # be careful with pointers here, or we have trouble
                            # with garbage collection
                            data = np.copy(ordered_data[shuffle_idx])
                            del ordered_data
                            order_data = None
                            self.current_data[path] = data
                        else:
                            self.current_data[path] = ordered_data

                shuffle_idx = None
        else:
            shuffle_idx = None
            for path in current_paths:
                if path not in self.current_data:
                    ordered_data = load_data(path)
                    self.cache_order.append(path)
                    if shuffle_idx is None and self.randomize:
                        shuffle_idx = np.arange(ordered_data.shape[0])
                        self.rdm.shuffle(shuffle_idx)

                    if self.randomize:
                        # be careful with pointers here, or we have trouble
                        # with garbage collection
                        data = np.copy(ordered_data[shuffle_idx])
                        del ordered_data
                        order_data = None
                        self.current_data[path] = data
                    else:
                        self.current_data[path] = ordered_data

    def create_batch_parts(self, current_paths, start, end):
        # index loaded data for minibatch
        batch_parts = []
        if isinstance(current_paths[0], list):
            start = start[0]
            end = end[1]
            for i in range(len(current_paths[0])):
                x1 = self.current_data[current_paths[0][i]][start:]
                x2 = self.current_data[current_paths[1][i]][:end]
                if len(x1.shape) == 1:
                    x = np.hstack([x1, x2])
                else:
                    x = np.vstack([x1, x2])
                batch_parts.append(x)
        else:
            for path in current_paths:
                batch_parts.append(self.current_data[path][start:end])

        return batch_parts

    def determine_cache_size(self):
        total_bytes = 0
        for path, shard in self.current_data.items():
            total_bytes += shard.nbytes
        return total_bytes/(1024.0**3.0)

    def clean_cache(self, current_paths):
        # delete unused cached data
        i = 0
        n = len(self.cache_order)
        while i < n:
            if self.cache_order[i] in current_paths:
                i += 1
                continue
            path = self.cache_order.pop(i)
            self.current_data.pop(path, None)
            GB_usage = self.determine_cache_size()
            n -= 1
            if GB_usage < self.cache_size_GB: break

    def publish_at_prepared_batch_event(self, batch_parts):
        for i, obs in enumerate(self.stream_batcher.at_batch_prepared_observers):
            self.t.tick(str(i))
            batch_parts = obs.at_batch_prepared(batch_parts)
            self.t.tick(str(i))
        return batch_parts

    def run(self):
        while not self.stopped():

            # we have this to terminate threads gracefully
            # if we use daemons then the terminational signal might not be heard while loading files
            # thus causing ugly exceptions
            try:
                batch_idx = self.stream_batcher.work.get(block=False, timeout=1.0)
            except:
                continue

            if self.randomize:
                n = 0
                while (n - self.stream_batcher.batch_size + 1) <= 0:
                    shard_idx = self.rdm.choice(len(list(self.shard2batchidx.keys())), 1, p=self.shard_fractions)[0]
                    current_paths = self.paths[shard_idx]

                    self.load_files_if_needed(current_paths)

                    n = self.current_data[current_paths[0]].shape[0]
                start = self.rdm.randint(0, n-self.stream_batcher.batch_size+1)
                end = start + self.stream_batcher.batch_size

                batch_parts = self.create_batch_parts(current_paths, start, end)
            else:
                if batch_idx not in self.batchidx2paths:
                    log.error('{0}, {1}', batch_idx, list(self.batchidx2paths.keys()))
                current_paths = self.batchidx2paths[batch_idx]
                start, end = self.batchidx2start_end[batch_idx]

                self.load_files_if_needed(current_paths)
                batch_parts = self.create_batch_parts(current_paths, start, end)


            batch_parts = self.publish_at_prepared_batch_event(batch_parts)
            # pass data to streambatcher
            self.stream_batcher.prepared_batches[batch_idx] = batch_parts
            try:
                self.stream_batcher.prepared_batchidx.put(batch_idx, block=False, timeout=1.0)
            except:
                continue

            GB_usage = self.determine_cache_size()
            if GB_usage > self.cache_size_GB:
                self.clean_cache(current_paths)
            self.batches_processes += 1
            if self.batches_processes % 100 == 0:
                if benchmark:
                    for i, obs in enumerate(self.stream_batcher.at_batch_prepared_observers):
                        t = self.t.tock(str(i))


class StreamBatcher(object):
    def __init__(self, pipeline_name, name, batch_size, loader_threads=4, randomize=False, seed=None, keys=['input', 'support', 'target'], is_volatile=False, cache_size_GB=4):
        config_path = join(get_data_path(), pipeline_name, name, 'hdf5_config.pkl')
        if not exists(config_path):
            log.error('Path {0} does not exists! Have you forgotten to preprocess your dataset?', config_path)
        config = pickle.load(open(config_path, 'rb'))
        self.paths = config['paths']
        self.fractions = config['fractions']
        self.num_batches = int(np.sum(config['counts']) / batch_size)
        self.max_lengths = config['max_lengths']
        self.batch_size = batch_size
        self.batch_idx = 0
        self.prefetch_batch_idx = 0
        self.loaders = []
        self.prepared_batches = {}
        self.prepared_batchidx = queue.Queue()
        self.work = queue.Queue()
        self.cached_batches = {}
        self.end_iter_observers = []
        self.end_epoch_observers = []
        self.start_epoch_observers = []
        self.at_batch_prepared_observers = []
        self.state = BatcherState()
        self.current_iter = 0
        self.current_epoch = 0
        self.timer = Timer()
        self.loader_threads = loader_threads
        if Config.backend == Backends.TORCH:
            from spodernet.backends.torchbackend import TorchConverter, TorchCUDAConverter
            self.subscribe_to_batch_prepared_event(DictConverter(keys))
            self.subscribe_to_batch_prepared_event(TorchConverter(is_volatile))
            if Config.cuda:
                import torch
                self.subscribe_to_batch_prepared_event(TorchCUDAConverter(torch.cuda.current_device()))
        elif Config.backend == Backends.TENSORFLOW:
            from spodernet.backends.tfbackend import TensorFlowConverter
            self.subscribe_to_batch_prepared_event(TensorFlowConverter())
        elif Config.backend == Backends.TEST:
            pass
        elif Config.backend == Backends.CNTK:
            self.subscribe_to_batch_prepared_event(DictConverter(keys))
        else:
            raise Exception('Backend has unsupported value {0}'.format(Config.backend))


        batchidx2paths, batchidx2start_end, shard2batchidx = self.create_batchidx_maps(config['counts'])

        for i in range(loader_threads):
            seed = 2345 + (i*83)
            self.loaders.append(DataLoaderSlave(self, batchidx2paths, batchidx2start_end, randomize, self.paths, shard2batchidx, seed, self.fractions, cache_size_GB))
            self.loaders[-1].start()


    def __del__(self):
        log.debug('Stopping threads...')
        for worker in self.loaders:
            worker.stop()

        log.debug('Waiting for threads to finish...')
        while threading.active_count() > 0:
            time.sleep(0.1)

    def subscribe_end_of_iter_event(self, observer):
        self.end_iter_observers.append(observer)

    def subscribe_end_of_epoch_event(self, observer):
        self.end_epoch_observers.append(observer)

    def subscribe_to_events(self, observer):
        self.subscribe_end_of_iter_event(observer)
        self.subscribe_end_of_epoch_event(observer)

    def subscribe_to_batch_prepared_event(self, observer):
        self.at_batch_prepared_observers.append(observer)

    def subscribe_to_start_of_epoch_event(self, observer):
        self.start_epoch_observers.append(observer)

    def publish_end_of_iter_event(self):
        self.state.current_idx = self.batch_idx
        self.state.current_epoch = self.current_epoch
        self.state.num_batches = self.num_batches

        if self.batch_idx == 0:
            self.current_iter += 1
            for obs in self.start_epoch_observers:
                obs.at_start_of_epoch_event(self.state)
            return
        for obs in self.end_iter_observers:
            obs.at_end_of_iter_event(self.state)
        self.state.clear()
        self.current_iter += 1

    def publish_end_of_epoch_event(self):
        self.state.current_idx = self.batch_idx
        self.state.current_epoch = self.current_epoch
        self.state.num_batches = self.num_batches
        self.state.timer = self.timer
        for obs in self.end_epoch_observers:
            obs.at_end_of_epoch_event(self.state)
        self.state.clear()
        self.current_epoch += 1

    def create_batchidx_maps(self, counts):
        counts_cumulative = np.cumsum(counts)
        counts_cumulative_offset = np.cumsum([0] + counts)
        batchidx2paths = {}
        batchidx2start_end = {}
        shard2batchidx = { 0 : []}
        paths = self.paths
        file_idx = 0
        for i in range(self.num_batches):
            start = i*self.batch_size
            end = (i+1)*self.batch_size
            if end > counts_cumulative[file_idx] and file_idx+1 < len(paths):
                start_big_batch = start - counts_cumulative_offset[file_idx]
                end_big_batch = end - counts_cumulative_offset[file_idx+1]
                batchidx2start_end[i] = ((start_big_batch, None), (None, end_big_batch))
                batchidx2paths[i] = (paths[file_idx], paths[file_idx+1])

                shard2batchidx[file_idx].append(i)
                file_idx += 1
                shard2batchidx[file_idx] = [i]
            else:
                start_big_batch = start - counts_cumulative_offset[file_idx]
                end_big_batch = end - counts_cumulative_offset[file_idx]
                batchidx2start_end[i] = (start_big_batch, end_big_batch)
                batchidx2paths[i] = paths[file_idx]
                shard2batchidx[file_idx].append(i)

        return batchidx2paths, batchidx2start_end, shard2batchidx


    def get_next_batch_parts(self):
        if self.batch_idx in self.cached_batches:
            return self.cached_batches.pop(self.batch_idx)
        else:
            batch_idx = self.prepared_batchidx.get()
            if self.batch_idx == batch_idx:
                return self.prepared_batches.pop(self.batch_idx)
            else:
                if batch_idx in self.prepared_batches:
                    self.cached_batches[batch_idx] = self.prepared_batches.pop(batch_idx)
                return self.get_next_batch_parts()

    def __iter__(self):
        return self


    def __next__(self):
        if self.batch_idx == 0:
            while self.prefetch_batch_idx < self.loader_threads:
                self.work.put(self.prefetch_batch_idx)
                self.prefetch_batch_idx += 1
        if self.batch_idx < self.num_batches:
            batch_parts = self.get_next_batch_parts()
            self.publish_end_of_iter_event()

            self.batch_idx += 1
            self.work.put(self.prefetch_batch_idx)
            self.prefetch_batch_idx +=1
            if self.prefetch_batch_idx >= self.num_batches:
                self.prefetch_batch_idx = 0

            return batch_parts
        else:
            self.batch_idx = 0
            self.publish_end_of_epoch_event()
            raise StopIteration()

    next = __next__
