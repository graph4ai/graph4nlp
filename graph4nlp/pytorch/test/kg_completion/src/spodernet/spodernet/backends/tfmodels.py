import tensorflow as tf
from tensorflow import placeholder
from spodernet.backends.tfbackend import TensorFlowConfig
from spodernet.utils.global_config import Config
from spodernet.frontend import AbstractModel
import numpy as np

def reader(inputs, lengths, output_size, contexts=(None, None), scope=None):
    with tf.variable_scope(scope or "reader") as varscope:

        cell = tf.contrib.rnn.LSTMCell(output_size, state_is_tuple=True,initializer=tf.contrib.layers.xavier_initializer())

        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0-Config.dropout)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            inputs,
            sequence_length=lengths,
            initial_state_fw=contexts[0],
            initial_state_bw=contexts[1],
            dtype=tf.float32)

        return outputs, states

def predictor(inputs, targets, target_size):
    init = tf.contrib.layers.xavier_initializer(uniform=True) #uniform=False for truncated normal
    logits = tf.contrib.layers.fully_connected(inputs, target_size, weights_initializer=init, activation_fn=None)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
            labels=targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')
    return [logits, loss, predict]


class TFEmbedding(AbstractModel):

    def __init__(self, embedding_size, num_embeddings, scope=None):
        super(TFEmbedding, self).__init__()

        self.embedding_size = embedding_size
        self.scope = scope
        self.num_embeddings = num_embeddings

    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, ['input', 'support'])
        self.expected_args('None', 'None')
        self.generated_outputs('input idx, support idx', 'both sequences have shape = [batch, timesteps, embedding dim]')

        embeddings = tf.get_variable("embeddings", [self.num_embeddings, self.embedding_size],
                                initializer=tf.random_normal_initializer(0., 1./np.sqrt(self.embedding_size)),
                                trainable=True, dtype="float32")

        with tf.variable_scope("embedders") as varscope:
            seqQ = tf.nn.embedding_lookup(embeddings, TensorFlowConfig.inp)
            varscope.reuse_variables()
            seqS = tf.nn.embedding_lookup(embeddings, TensorFlowConfig.support)

        return seqQ, seqS

class TFPairedBiDirectionalLSTM(AbstractModel):

    def __init__(self, hidden_size, scope=None, conditional_encoding=True):
        super(TFPairedBiDirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.scope = scope
        if not conditional_encoding:
            raise NotImplementedError("conditional_encoding=False is not implemented yet.")

    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, ['input_length', 'support_length'])
        self.expected_args('seq input, seq support', 'dimension of both: [batch, timesteps, embedding dim]')
        self.generated_outputs('stacked outputs of last timestep', 'dim is [batch_size, 2x hidden size]')

        seqQ, seqS = args

        with tf.variable_scope(self.scope or "conditional_reader_seq1") as varscope1:
            #seq1_states: (c_fw, h_fw), (c_bw, h_bw)
            _, seq1_states = reader(seqQ, TensorFlowConfig.input_length, self.hidden_size, scope=varscope1)
        with tf.variable_scope(self.scope or "conditional_reader_seq2") as varscope2:
            varscope1.reuse_variables()
            # each [batch_size x max_seq_length x output_size]
            outputs, states = reader(seqS, TensorFlowConfig.support_length, self.hidden_size, seq1_states, scope=varscope2)

        output = tf.concat([states[0][1], states[1][1]], 1)

        return [output]

class TFSoftmaxCrossEntropy(AbstractModel):

    def __init__(self, num_labels):
        super(TFSoftmaxCrossEntropy, self).__init__()
        self.num_labels = num_labels

    def forward(self, str2var, *args):
        self.expected_str2var_keys(str2var, ['target'])
        self.expected_args('some inputs', 'dimension: [batch, any]')
        self.generated_outputs('logits, loss, argmax', 'dimensions: logits = [batch, labels], loss = 1x1, argmax = [batch, 1]')
        outputs_prev_layer = args[0]

        logits, loss, argmax = predictor(outputs_prev_layer, TensorFlowConfig.target, self.num_labels)

        return [logits, loss, argmax]
