# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 13:26:23 2017

@author: whisp
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import embedding_rnn_seq2seq
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl


#generate random input
def gen(input_size):
    st = ""
    for i in range(input_size):
        x = np.random.randint(0, 255)
        x_ = np.binary_repr(x, width=8)
        st = st + " " + x_
    return st[1:]

# Convert char to dataframe
def convert_dataframe(st_, input_size):
    df1 = pd.DataFrame()
    for i in range(input_size):
        df1[i] = np.array(list(st_.split(" ")[i])).astype(np.int)
    return df1

#plot figure
def draw(df):
    # Set up the matplotlib figure
    mask = np.zeros_like(df, dtype=np.bool)
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and input bits
    sns.heatmap(df, mask=mask, cmap="YlGnBu", vmax=1.5, vmin=-0.1,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


input_size = 2000
word_lenght = 20
st = gen(input_size)
df = convert_dataframe(st, input_size)
df1 = np.array(df)
#draw(df)
    
# Parameters
learning_rate = 0.001
training_iters = 100
batch_size = 2
display_step = 10

# Network Parameters
n_input = 8 # data input
n_hidden = 5 # hidden layer num of features


# tf Graph input
x = tf.placeholder("int32", [None, word_lenght, n_input])
y = tf.placeholder("int32", [None, n_input])   
with tf.variable_scope("train_test", reuse = None):
    x = tf.unstack(x, word_lenght, 1)
    outputs, states = embedding_rnn_seq2seq(encoder_inputs = x, 
                                        decoder_inputs = [0]*20, 
                                        cell = core_rnn_cell_impl.LSTMCell(n_hidden),
                                        num_encoder_symbols = 256, 
                                        num_decoder_symbols = 256,
                                        embedding_size = 100, 
                                        output_projection=None,
                                        feed_previous=False)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)    
    
# Evaluate model
correct_pred = tf.equal(tf.argmax(outputs,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x = df1[:,(step-1) * batch_size*20 : step * batch_size*20].reshape(-1, word_lenght, n_input)
        batch_y = batch_x
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

#    # Calculate accuracy for 128 mnist test images
#    test_len = 128
#    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
#    test_label = mnist.test.labels[:test_len]
#    print "Testing Accuracy:", \
#        sess.run(accuracy, feed_dict={x: test_data, y: test_label})

  
    




















