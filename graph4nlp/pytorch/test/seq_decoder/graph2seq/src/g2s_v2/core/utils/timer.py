import time


class Timer(object):
    """Computes elapsed time."""
    def __init__(self, name):
        self.name = name
        self.running = True
        self.total = 0
        self.start = round(time.time(), 2)
        self.intervalTime = round(time.time(), 2)
        print("<> <> <> Starting Timer [{}] <> <> <>".format(self.name))

    def reset(self):
        self.running = True
        self.total = 0
        self.start = round(time.time(), 2)
        return self

    def interval(self, intervalName=''):
        intervalTime = self._to_hms(round(time.time() - self.intervalTime, 2))
        print("<> <> Timer [{}] <> <> Interval [{}]: {} <> <>".format(self.name, intervalName, intervalTime))
        self.intervalTime = round(time.time(), 2)
        return intervalTime

    def stop(self):
        if self.running:
            self.running = False
            self.total += round(time.time() - self.start, 2)
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = round(time.time(), 2)
        return self

    def time(self):
        if self.running:
            return round(self.total + time.time() - self.start, 2)
        return self.total

    def finish(self):
        if self.running:
            self.running = False
            self.total += round(time.time() - self.start, 2)
            elapsed = self._to_hms(self.total)
        print("<> <> <> Finished Timer [{}] <> <> <> Total time elapsed: {} <> <> <>".format(self.name, elapsed))

    def _to_hms(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%dh %02dm %02ds" % (h, m, s)
