#These are completly useless, but they signify intent which is important.

class IAtIterEndObservable(object):
    def at_end_of_iter_event(self, batcher_state):
        raise NotImplementedError('Subclasses of IAtIterEndObservable need to override the end_of_iter_event method')

class IAtEpochStartObservable(object):
    def at_start_of_epoch_event(self, batcher_state):
        raise NotImplementedError('Subclasses of IAtEpochStartObservable need to override the at_start_of_epoch method')

class IAtEpochEndObservable(object):
    def at_end_of_epoch_event(self, batcher_state):
        raise NotImplementedError('Subclasses of IAtEpochEndObservable need to override the end_of_iter_epoch method')

class IAtBatchPreparedObservable(object):
    def at_batch_prepared(self, batch_parts):
        raise NotImplementedError('Subclasses of IAtBatchPreparedObservable need to override the at_batch_prepared method')
