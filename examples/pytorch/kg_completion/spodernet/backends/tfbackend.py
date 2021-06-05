import tensorflow as tf

from spodernet.interfaces import IAtBatchPreparedObservable
from spodernet.utils.util import Timer
from spodernet.utils.global_config import Config

class TensorFlowConfig:
    inp = None
    support = None
    input_length = None
    support_length = None
    target = None
    index = None
    sess = None

    @staticmethod
    def init_batch_size(batch_size):
        TensorFlowConfig.inp = tf.placeholder(tf.int64, [batch_size, None])
        TensorFlowConfig.support = tf.placeholder(tf.int64, [batch_size, None])
        TensorFlowConfig.input_length = tf.placeholder(tf.int64, [batch_size,])
        TensorFlowConfig.support_length = tf.placeholder(tf.int64, [batch_size,])
        TensorFlowConfig.target = tf.placeholder(tf.int64, [batch_size])
        TensorFlowConfig.index = tf.placeholder(tf.int64, [batch_size])

    @staticmethod
    def get_session():
        if TensorFlowConfig.sess is None:
            TensorFlowConfig.sess = tf.Session()
        return TensorFlowConfig.sess



class TensorFlowConverter(IAtBatchPreparedObservable):

    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts
        if TensorFlowConfig.inp == None:
            log.error('You need to initialize the batch size via TensorflowConfig.init_batch_size(batchsize)!')
        feed_dict = {}
        feed_dict[TensorFlowConfig.inp] = inp
        feed_dict[TensorFlowConfig.support] = sup
        feed_dict[TensorFlowConfig.input_length] = inp_len
        feed_dict[TensorFlowConfig.support_length] = sup_len
        feed_dict[TensorFlowConfig.target] = t
        feed_dict[TensorFlowConfig.index] = idx

        str2var = {}
        str2var['input'] = TensorFlowConfig.inp
        str2var['input_length'] = TensorFlowConfig.input_length
        str2var['support'] = TensorFlowConfig.support
        str2var['support_length'] = TensorFlowConfig.support_length
        str2var['target'] = TensorFlowConfig.target
        str2var['index'] = TensorFlowConfig.index

        return str2var, feed_dict

def build_str2var_dict():
    str2var = {}
    if TensorFlowConfig.inp is not None:
        str2var['input'] = TensorFlowConfig.inp
    if TensorFlowConfig.support is not None:
        str2var['support'] = TensorFlowConfig.support
    if TensorFlowConfig.target is not None:
        str2var['target'] = TensorFlowConfig.target
    if TensorFlowConfig.input_length is not None:
        str2var['input_length'] = TensorFlowConfig.input_length
    if TensorFlowConfig.support_length is not None:
        str2var['support_length'] = TensorFlowConfig.support_length
    if TensorFlowConfig.index is not None:
        str2var['index'] = TensorFlowConfig.index
        return str2var

class TFTrainer(object):
    def __init__(self, model):
        self.sess = TensorFlowConfig.get_session()
        str2var = build_str2var_dict()
        self.logits, self.loss, self.argmax = model.forward(str2var)
        optimizer = tf.train.AdamOptimizer(0.001)

        if Config.L2 != 0.0:
            self.loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * Config.L2

        self.min_op = optimizer.minimize(self.loss)

        tf.global_variables_initializer().run(session=self.sess)

    def train_model(self, batcher, epochs=1, iterations=None):
        for epoch in range(epochs):
            for i, (str2var, feed_dict) in enumerate(batcher):
                _, argmax_values = self.sess.run([self.min_op, self.argmax], feed_dict=feed_dict)

                batcher.state.argmax = argmax_values
                batcher.state.targets = feed_dict[TensorFlowConfig.target]

                if iterations > 0:
                    if i == iterations: break

    def eval_model(self, batcher, iterations=None):
        for i, (str2var, feed_dict) in enumerate(batcher):
            argmax_values = self.sess.run([self.argmax], feed_dict=feed_dict)[0]

            batcher.state.argmax = argmax_values
            batcher.state.targets = feed_dict[TensorFlowConfig.target]

            if iterations > 0:
                if i == iterations: break

