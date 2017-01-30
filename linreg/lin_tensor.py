from __future__ import print_function

import tensorflow as tf
import pydub as pd
import numpy as np
rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
audio_X = pd.AudioSegment.from_mp3("lq_spaceoddity.mp3")
audio_Y = pd.AudioSegment.from_mp3("hq_spaceoddity.mp3")

samples_X = np.asarray(audio_X.get_array_of_samples())
samples_Y = np.asarray(audio_Y.get_array_of_samples())

split = int(np.round(samples_X.size * .6))

train_X = samples_X[0:split]
test_X = samples_X[split:]
train_Y = samples_Y[0:split]
test_Y = samples_X[split:]

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    pred_Y = sess.run(pred, feed_dict={X: test_X})
