# Lab 6-1 : Q-Network for Cart Pole
import gym
import numpy as np
import tensorflow.compat.v1 as tf # To use tensorflow 1.x
import matplotlib.pyplot as plt
tf.disable_v2_behavior() # Make tensorflow 2.x's behavior disable

env = gym.make('CartPole-v0')

# Constants defining neural network
learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

X = tf.placeholder(tf.float32, [None, input_size], name='input_x')

# First layer of weights
W1 = tf.get_variable('W1', shape=[input_size, output_size],
                     initializer=tf.random_uniform_initializer())
Qpred = tf.matmul(X, W1)

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

# Loss function
loss = tf.reduce_sum(tf.square(Y - Qpred))
# Learning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# values for Q leaerning
num_episodes = 2000
dis = 0.9
rList = []

# Setting up our environment
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
    e = 1./((i / 10) + 1)
    rAll = 0
    step_count = 0
    s = env.reset()
    done = False

    # the Q-Network training
    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size])
        # Choose an action by greedily (with e chance of random action) from the Q-network
        Qs = sess.run(Qpred, feed_dict={X: x})
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        # Get new state and reward from environment
        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0, a] = -100
        else:
            x1 = np.reshape(s1, [1, input_size])
            # Obtain the Q' values by feeding the new state through network
            Qs1 = sess.run(Qpred, feed_dict={X: x1})
            Qs[0, a] = reward + dis * np.max(Qs1)

        # Train network using target and predicted Q values on each episode
        sess.run(train, feed_dict={X: x, Y: Qs})
        s = s1

    rList.append(step_count)
    print("Episode: {} steps: {}".format(i, step_count))
    # If last 10's avg steps are 500, it's good enough
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break

# See our trained network in action
observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = np.reshape(observation, [1, input_size])
    Qs = sess.run(Qpred, feed_dict={X: x})
    a = np.argmax(Qs)

    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break