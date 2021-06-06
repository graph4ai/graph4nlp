import numpy as np
import scipy.stats
import datetime

from spodernet.interfaces import IAtIterEndObservable, IAtEpochEndObservable, IAtEpochStartObservable
from spodernet.utils.util import Timer
from spodernet.utils.global_config import Config, Backends

from spodernet.utils.logger import Logger
log = Logger('hooks.py.txt')

class AbstractHook(IAtIterEndObservable, IAtEpochEndObservable):
    def __init__(self, name, metric_name, print_every_x_batches):
        self.epoch_errors = []
        self.current_scores = []
        self.name = name
        self.iter_count = 0
        self.print_every = print_every_x_batches
        self.metric_name = metric_name
        self.epoch = 1

        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        self.n = 0
        self.epoch_n = 0
        self.mean = 0
        self.M2 = 0
        self.load_backend_specific_functions()

    def load_backend_specific_functions(self):
        if Config.backend == Backends.TORCH:
            from torch.autograd import Variable
            def convert_state(state):
                if isinstance(state.targets, Variable):
                    state.targets = state.targets.data
                if isinstance(state.argmax, Variable):
                    state.argmax = state.argmax.data
                if isinstance(state.pred, Variable):
                    state.pred = state.pred.data
                if isinstance(state.loss, Variable):
                    state.loss = state.loss.data
                if isinstance(state.multi_labels, Variable):
                    state.multi_labels = state.multi_labels.data

                return state

            self.convert_state = convert_state
        else:
            self.convert_state = lambda x: x

    def calculate_metric(self, state):
        raise NotImplementedError('Classes that inherit from abstract hook need to implement the calcualte metric method.')

    def at_end_of_iter_event(self, state):
        state = self.convert_state(state)
        metric = self.calculate_metric(state)
        #print(metric)

        self.n += 1
        delta = metric - self.mean
        self.mean += delta/self.n
        delta2 = metric - self.mean
        self.M2 += delta*delta2

        self.current_scores.append(metric)
        self.iter_count += 1
        if self.iter_count % self.print_every == 0:
            lower, upper, m, n = self.print_statistic()
            self.n = 0
            self.mean = 0
            self.M2 = 0
            return lower, upper, m, n
        return 0, 0, self.mean, self.n

    def at_end_of_epoch_event(self, state):
        if self.n == 0: return 0, 0, 0, 0
        self.epoch_errors.append(self.get_confidence_intervals())
        lower, upper, m, n = self.print_statistic(True)
        del self.current_scores[:]
        self.n = 0
        self.mean = 0
        self.M2 = 0
        self.epoch += 1
        self.iter_count = 0
        return lower, upper, m, n

    def get_confidence_intervals(self, percentile=0.99, limit=1000):
        z = scipy.stats.norm.ppf(percentile)
        var = self.M2/ (self.n)
        SE = np.sqrt(var/self.n)
        lower = self.mean-(z*SE)
        upper = self.mean+(z*SE)
        return [self.n, lower, self.mean, upper]

    def print_statistic(self, at_epoch_end=False):
        n, lower, m, upper = self.get_confidence_intervals()
        str_message = '{3} {4}: {2:.5}\t99% CI: ({0:.5}, {1:.5}), n={5}'.format(lower, upper, m, self.name, self.metric_name, self.n)
        if at_epoch_end: log.info('\n')
        if at_epoch_end: log.info('#'*40)
        if at_epoch_end: log.info(' '*10 + 'COMPLETED EPOCH: {0}'.format(self.epoch) + ' '*30)
        log.info(str_message)
        if at_epoch_end: log.info('#'*40)
        if at_epoch_end: log.info('\n')
        return lower, upper, m, n


class AccuracyHook(AbstractHook):
    def __init__(self, name='', print_every_x_batches=1000):
        super(AccuracyHook, self).__init__(name, 'Accuracy', print_every_x_batches)
        self.func = None
        self.topk = 1
        if Config.backend == Backends.TORCH:
            import torch
            self.func = lambda x: torch.sum(x)

    def calculate_metric(self, state):
        if Config.backend == Backends.TORCH:
            correct = 0.0
            if len(state.argmax.size()) == 1:
                correct += self.func(state.targets==state.argmax)
            else:
                topk = state.argmax.size(1)
                for i in range(topk):
                    correct += self.func(state.targets==state.argmax[:, i])
            n = state.argmax.size()[0]
            return correct.item()/np.float32(n)
        elif Config.backend == Backends.TENSORFLOW:
            n = state.argmax.shape[0]
            return np.sum(state.targets==state.argmax)/np.float32(n)
        elif Config.backend == Backends.TEST:
            n = state.argmax.shape[0]
            return np.sum(state.targets==state.argmax)/np.float32(n)
        else:
            raise Exception('Backend has unsupported value {0}'.format(Config.backend))


class TopKRankingLoss(AbstractHook):
    def __init__(self, k, filtered=False, name='', print_every_x_batches=1000):
        super(TopKRankingLoss, self).__init__(name, '{1}Hits@{0} loss'.format(k, ('' if not filtered else 'Filtered ')), print_every_x_batches)
        self.func = None
        self.argsort = None
        self.sum_func = None
        self.k = k
        self.filtered = filtered
        if Config.backend == Backends.TORCH:
            import torch
            self.argsort = lambda x, k: torch.topk(x, k)
            self.sum_func = lambda x: torch.sum(x)


    def calculate_metric(self, state):
        if Config.backend == Backends.TORCH:
            if self.filtered:
                import torch
                saved = torch.index_select(state.pred,1,state.targets)
                state.pred[state.multi_labels.byte()] = -100000.0
                state.pred.index_copy_(1, state.targets, saved)

            max_values, argmax = self.argsort(state.pred, self.k)
            in_topk = 0
            for i in range(self.k):
                in_topk += self.sum_func(argmax[:,i] == state.targets)
            n = state.pred.size()[0]
            return in_topk/np.float32(n)
        else:
            raise Exception('Backend has unsupported value {0}'.format(Config.backend))



class LossHook(AbstractHook):
    def __init__(self, name='', print_every_x_batches=1000):
        super(LossHook, self).__init__(name, 'Loss', print_every_x_batches)

    def calculate_metric(self, state):
        if Config.backend == Backends.TORCH:
            state = self.convert_state(state)
            return state.loss.item()
        else:
            return state.loss


class IntersectionHook(AbstractHook):
    def __init__(self, name='', print_every_x_batches=1000):
        super(IntersectionHook, self).__init__(name, 'Intersection', print_every_x_batches)

    def calculate_metric(self, state):
        state = self.convert_state(state)
        preds = state.pred
        targets = state.targets
        if Config.cuda:
            preds = preds.cpu()
            targets = targets.cpu()

        preds = preds.numpy()
        targets = targets.numpy()
        n = targets.size
        k = 0
        for row in range(Config.batch_size):
            k += np.intersect1d(preds[row], targets[row]).size

        return k/float(n)



class ETAHook(AbstractHook, IAtEpochStartObservable):
    def __init__(self, name='', print_every_x_batches=1000):
        super(ETAHook, self).__init__(name, 'ETA', print_every_x_batches)
        self.t = Timer(silent=True)
        self.cumulative_t = 0.0
        self.skipped_first = False

    def get_time_string(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h < 0: h = 0
        if m < 0: m = 0
        if s < 0: s = 0
        return "%d:%02d:%02d" % (h, m, s)

    def calculate_metric(self, state):
        n = state.num_batches
        i = state.current_idx
        cumulative_t = self.t.tick('ETA')
        total_time_estimate = (cumulative_t/i)*n
        self.t.tick('ETA')
        self.cumulative_t = cumulative_t

        return total_time_estimate

    def print_statistic(self):
        if not self.skipped_first:
            # the first estimation is very unreliable for time measures
            self.skipped_first = True
            return 0, 0, 0, 0
        n, lower, m, upper = self.get_confidence_intervals()
        lower -= self.cumulative_t
        m -= self.cumulative_t
        upper -= self.cumulative_t
        lower, m, upper = self.get_time_string(lower), self.get_time_string(m), self.get_time_string(upper)
        log.info('{3} {4}: {2}\t99% CI: ({0}, {1}), n={5}'.format(lower, upper, m, self.name, self.metric_name, n))
        return lower, upper, m, n

    def at_start_of_epoch_event(self, batcher_state):
        self.t.tick('ETA')
        t = self.t.tick('Epoch')

    def at_end_of_epoch_event(self, state):
        self.t.tock('ETA')
        epoch_time = self.t.tock('Epoch')
        self.epoch_errors.append([epoch_time])
        log.info('Total epoch time: {0}'.format(self.get_time_string(epoch_time)))
        del self.current_scores[:]
        self.n = 0
        self.mean = 0
        self.M2 = 0
        self.skipped_first = False
        self.epoch += 1
        return epoch_time
