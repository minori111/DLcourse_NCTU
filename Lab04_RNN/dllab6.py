import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_seq2seq ,sequence_loss
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.framework import ops

sess = tf.InteractiveSession()
seq_length = 5
batch_size = 64
vocab_size = 10
embedding_dim = 100
memory_dim = 500
pad = 5
enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,),
                        name="labels%i" % t)
          for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="de_inp%i" % t)
           for t in range(seq_length)]

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))
cell = LSTMCell(memory_dim)

dec_outputs, dec_memory = embedding_rnn_seq2seq(
    enc_inp, dec_inp, cell, vocab_size+1, vocab_size+2,embedding_size = embedding_dim,feed_previous=False)
loss = sequence_loss(dec_outputs, labels, weights, vocab_size)
tf.summary.scalar("loss", loss)
magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
tf.summary.scalar("magnitude at t=1", magnitude)
#summary_op =  tf.merge_all_summaries()
learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)
sess.run(tf.initialize_all_variables())
def train_batch(batch_size):
    X = [np.random.choice(vocab_size, size=(seq_length,), replace=False)+1
         for _ in range(batch_size)]
    Y = X[:]
    
    # Dimshuffle to seq_len * batch_size
    X = np.array(X).T
    Y = np.array(Y).T

    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
    feed_dict.update({dec_inp[t]: np.full(batch_size,11) for t in range(seq_length)})
    loss_t, summary = sess.run([train_op, loss], feed_dict)
    return loss_t, summary
for t in range(1000):
    _, loss_t = train_batch(batch_size)
    print(t)
X_batch = [np.random.choice(vocab_size, size=seq_length,replace=False)+1
           for _ in range(10)]
X_batch = np.array(X_batch).T

feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
feed_dict.update({dec_inp[t]: np.full(10,11) for t in range(seq_length)})   
dec_outputs_batch = sess.run(dec_outputs, feed_dict)
    
print(np.array(X_batch).T)
print(np.array([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]).T)