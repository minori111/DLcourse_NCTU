# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:46:42 2017

@author: whisp
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_seq2seq ,sequence_loss
from tensorflow.contrib.rnn import LSTMCell


tf.reset_default_graph()
sess = tf.InteractiveSession()
#seq_length = 5
#batch_size = 64
#vocab_size = 10
#embedding_dim = 100
#memory_dim = 100
#pad = 5
seq_length = 20
batch_size = 256

vocab_size = 256
embedding_dim = 100

memory_dim = 500

enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" % t)
           for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" % t)
          for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
           + enc_inp[:-1])

cell = LSTMCell(memory_dim)

dec_outputs, dec_memory = embedding_rnn_seq2seq(
    enc_inp, dec_inp, cell, vocab_size, vocab_size, 
    embedding_dim, feed_previous = False)

loss = sequence_loss(dec_outputs, labels, weights)

global_step = tf.Variable(0, trainable=False)
boundaries = [10000]
values = [0.01, 0.001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
#learning_rate = 0.01 #0.97485
#momentum = 0.9
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)


sess.run(tf.global_variables_initializer())

def train_batch(batch_size):
    X = [np.random.randint(0, vocab_size, size=(seq_length,))
         for _ in range(batch_size)]
    Y = X[:]
    
    # Dimshuffle to seq_len * batch_size
    X = np.array(X).T
    Y = np.array(Y).T

    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


for t in range(15000):
    loss_t= train_batch(batch_size)
    if t%100 == 0:
        print(loss_t.mean())


X_batch = [np.random.randint(0, vocab_size, size=(seq_length,))
           for _ in range(1000)]
X_batch = np.array(X_batch).T

feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
dec_outputs_batch = sess.run(dec_outputs, feed_dict)

print(np.array(X_batch))

print(np.array([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]))
print(np.mean(np.array(X_batch)==np.array([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch])))
sess.close()

