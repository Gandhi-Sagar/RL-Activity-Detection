import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from single_video_env import SingleVideoEnv
from single_video_env import action_space

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """ Forward-propagation. """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

env = SingleVideoEnv('serve.mp4', 32, 24)
acs = action_space()


x_size = 32*24
h_size = 256
y_size = 2

# Symbols
X = tf.placeholder("float", shape=[1, x_size])
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
num_episodes = 100

#create lists to contain total rewards
rList = []
correct = 0
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment = start video from the beginnning and get first new observation = first frame
        print('iteration: ' + str(i))
        s = env.reset()
        rAll = 0
        d = False
        while d != True:
            # Choose an action greedily (with e chance of random action) from the Q-network
            # feed_dict = {input_image_one_hot_encoded}
            # allQ has to be vector of 2, a is just max of that
            flatS = np.matrix(s.ravel())
            a, allQ = sess.run([predict, Qout], feed_dict={X: np.array(flatS)})
            if np.random.rand(1) < e:
                a[0] = acs.sample()
            # Get new state and reward from environment
            s1, r, d, correct = env.step(a[0])
            # Obtain the Q' values by feeding the new state through our network
            # feed_dict={X: np.identity(x_size)[s1:s1 + 1]} is again, one hot encoded next image
            if (d == False):
                flatS1 = np.matrix(s1.ravel())
                Q1 = sess.run(Qout, feed_dict={X: np.array(flatS1)})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + y * maxQ1
                # things are pretty much figured out till the line above
                # *mind you, we are never ever passing the known (from oracle) labels to the neural network
                # and ALL neural network learning is based on the rewardsd
                # Train our network using target and predicted Q values
                _ = sess.run([updateModel], feed_dict={X: np.array(flatS), nextQ: targetQ})
                rAll += r
                s = s1
            else:
                # Reduce chance of random action as we train the model.
                print('total correctly recognized: ' + str(correct))
                e = 1. / ((i / 50) + 10)
        rList.append(rAll)


print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
plt.plot(rList)
plt.show()


































