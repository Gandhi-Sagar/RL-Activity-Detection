import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from single_video_env import SingleVideoEnv

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """ Forward-propagation. """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

env = SingleVideoEnv('serve.mp4')


x_size = 32*24
h_size = 256
y_size = 2

# Symbols
X = tf.placeholder("float", shape=[None, x_size])
y = tf.placeholder("float", shape=[None, y_size])

# Weight initializations
w_1 = init_weights((x_size, h_size))
w_2 = init_weights((h_size, y_size))

# Forward propagation
Qout = forwardprop(X, w_1, w_2)
predict = tf.argmax(Qout, axis=1)

nextQ = tf.placeholder(shape=[1, 2], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Qout - nextQ))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 4000

#create lists to contain total rewards
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment = start video from the beginnning and get first new observation = first frame
        s = env.reset()
        rAll = 0
        d = False
        while d != True:
            # Choose an action greedily (with e chance of random action) from the Q-network
            # feed_dict = {input_image_one_hot_encoded}
            # allQ has to be vector of 2, a is just max of that
            a, allQ = sess.run([predict, Qout], feed_dict={X: np.identity(x_size)[s:s + 1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            # Get new state and reward from environment
            s1, r, d, _ = env.step(a[0])
            # Obtain the Q' values by feeding the new state through our network
            # feed_dict={X: np.identity(x_size)[s1:s1 + 1]} is again, one hot encoded next image
            Q1 = sess.run(Qout, feed_dict={X: np.identity(x_size)[s1:s1 + 1]})
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y * maxQ1
            # things are pretty much figured out till the line above
            # *mind you, we are never ever passing the known (from oracle) labels to the neural network
            # and ALL neural network learning is based on the rewardsd
            # Train our network using target and predicted Q values
            _ = sess.run([updateModel], feed_dict={X: np.identity(x_size)[s:s + 1], nextQ: targetQ})
            rAll += r
            s = s1
            if d == True:
                # Reduce chance of random action as we train the model.
                e = 1. / ((i / 50) + 10)
        rList.append(rAll)
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
plt.plot(rList)
plt.show()


































