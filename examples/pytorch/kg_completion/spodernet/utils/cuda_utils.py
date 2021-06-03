from __future__ import print_function
import torch
from torch.cuda import Event

class CUDATimer(object):
    def __init__(self, silent=False):
        self.cumulative_secs = {}
        self.current_ticks = {}
        self.silent = silent
        self.end = Event(enable_timing=True, blocking=True)

    def tick(self, name='default'):
        if name not in self.current_ticks:
            start = Event(enable_timing=True, blocking=True)
            start.record()
            self.current_ticks[name] = start

            return 0.0
        else:
            if name not in self.cumulative_secs:
                self.cumulative_secs[name] = 0
            self.end.record()
            self.end.synchronize()
            self.cumulative_secs[name] += self.current_ticks[name].elapsed_time(self.end)/1000.
            self.current_ticks.pop(name)

            return self.cumulative_secs[name]

    def tock(self, name='default'):
        self.tick(name)
        value = self.cumulative_secs[name]
        if not self.silent:
            print('Time taken for {0}: {1:.8f}s'.format(name, value))
        self.cumulative_secs.pop(name)
        if name in self.current_ticks:
            del self.current_ticks[name]
        self.current_ticks.pop(name, None)

        return value
