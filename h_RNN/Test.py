"""Short and sweet LSTM implementation in Tensorflow.
Motivation:
When Tensorflow was released, adding RNNs was a bit of a hack - it required
building separate graphs for every number of timesteps and was a bit obscure
to use. Since then TF devs added things like `dynamic_rnn`, `scan` and `map_fn`.
Currently the APIs are decent, but all the tutorials that I am aware of are not
making the best use of the new APIs.
Advantages of this implementation:
- No need to specify number of timesteps ahead of time. Number of timesteps is
  infered from shape of input tensor. Can use the same graph for multiple
  different numbers of timesteps.
- No need to specify batch size ahead of time. Batch size is infered from shape
  of input tensor. Can use the same graph for multiple different batch sizes.
- Easy to swap out different recurrent gadgets (RNN, LSTM, GRU, your new
  creative idea)
"""

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

map_fn = tf.map_fn


class MyLSTMCell(tf.contrib.rnn.BasicRNNCell):
    def __call__(self, x, state, scope="LSTM"):
        with tf.variable_scope(scope):
            s_old, h_old = tf.split(state, 2, 1)
            gates = layers.fully_connected(tf.concat([x, s_old], 1),
                                           num_outputs=4 * self._num_units,
                                           activation_fn=None)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            r1, g1, g2, g3 = tf.split(gates, 4, 1)
            r1, g1, g3 = tf.nn.sigmoid(r1), tf.nn.sigmoid(g1), tf.nn.sigmoid(g3)
            g2 = tf.nn.tanh(g2)
            h_new = (h_old * r1
                     + g1 * g2)
            s_new = tf.nn.tanh(h_new) * g3
            return s_new, tf.concat([s_new, h_new], 1)

    @property
    def state_size(self):
        return 2 * self._num_units


################################################################################
##                           DATASET GENERATION                               ##
##                                                                            ##
##  The problem we are trying to solve is adding two binary numbers. The      ##
##  numbers are reversed, so that the state of RNN can add the numbers        ##
##  perfectly provided it can learn to store carry in the state. Timestep t   ##
##  corresponds to bit len(number) - t.                                       ##
################################################################################

base = 10
NUM_BITS = 2
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 128
INPUT_SIZE = 3
RNN_HIDDEN = 32


def gen_seq(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % base)
        num //= base
    return res[::-1]


def generate_example(num_bits):
    ns = [int(random.randint(0, base ** num_bits - 1) / INPUT_SIZE) for _ in range(INPUT_SIZE)]
    res = sum(ns)
    return ([gen_seq(n, num_bits) for n in ns],
            gen_seq(res, num_bits))


def generate_batch(num_bits, batch_size):
    """Generates instance of a problem.
    Returns
    -------
    x: np.array
        two numbers to be added represented by bits.
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is one of [0,1] depending for first and
                second summand respectively
    y: np.array
        the result of the addition
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is always 0
    """
    x = np.empty((num_bits, batch_size, INPUT_SIZE))
    y = np.zeros((num_bits, batch_size, base))

    for i in range(batch_size):
        ns, r = generate_example(num_bits)
        for j in range(INPUT_SIZE):
            x[:, i, j] = ns[j]
        y[range(num_bits), i, r] = 1
    return x, y


################################################################################
##                           GRAPH DEFINITION                                 ##
################################################################################

OUTPUT_SIZE = base  # 1 bit per timestep
TINY = 1e-6  # to avoid NaNs in logs
LEARNING_RATE = 0.01

USE_LSTM = True

inputs = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))  # (time, batch, out)

## Here cell can be any function you want, provided it has two attributes:
#     - cell.zero_state(batch_size, dtype)- tensor which is an initial value
#                                           for state in __call__
#     - cell.__call__(input, state) - function that given input and previous
#                                     state returns tuple (output, state) where
#                                     state is the state passed to the next
#                                     timestep and output is the tensor used
#                                     for infering the output at timestep. For
#                                     example for LSTM, output is just hidden,
#                                     but state is memory + hidden
# Example LSTM cell with learnable zero_state can be found here:
#    https://gist.github.com/nivwusquorum/160d5cf7e1e82c21fad3ebf04f039317
if USE_LSTM:
    cell = MyLSTMCell(RNN_HIDDEN)
else:
    cell = tf.contrib.rnn.BasicRNNCell(RNN_HIDDEN)

# Create initial state. Here it is just a constant tensor filled with zeros,
# but in principle it could be a learnable parameter. This is a bit tricky
# to do for LSTM's tuple state, but can be achieved by creating two vector
# Variables, which are then tiled along batch dimension and grouped into tuple.
batch_size = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

# Given inputs (time, batch, input_size) outputs a tuple
#  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
#  - states:  (time, batch, hidden_size)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

# project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
# an extra layer here.
final_projection = lambda x: layers.fully_connected(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply projection to every timestep.
predicted_outputs = map_fn(final_projection, rnn_outputs)

# compute elementwise cross entropy.
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)

# optimize
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

################################################################################
##                           TRAINING LOOP                                    ##
################################################################################

valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epoch in range(1000):
    epoch_error = 0
    for _ in range(ITERATIONS_PER_EPOCH):
        x, y = generate_batch(num_bits=NUM_BITS+random.randint(0, 1), batch_size=BATCH_SIZE)
        epoch_error += session.run([error, train_fn], {
            inputs: x,
            outputs: y,
        })[0]
    epoch_error /= ITERATIONS_PER_EPOCH
    print("Epoch %d, train error: %.2f" % (epoch, epoch_error))
    x_test, y_test = generate_batch(num_bits=NUM_BITS+random.randint(0, 1), batch_size=1)
    ans = np.argmax(session.run(predicted_outputs, {
        inputs: x_test
    }), axis=2).ravel()
    x_test = x_test.astype(np.int)
    print("I think {} = {}, answer: {}...".format(
        " + ".join(
            ["".join(map(lambda n: str(n), x_test[..., 0, i])) for i in range(INPUT_SIZE)]
        ),
        "".join(map(lambda n: str(n), ans)),
        "".join(map(lambda n: str(n), np.argmax(y_test, axis=2).ravel())
    )))
