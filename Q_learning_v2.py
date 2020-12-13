# Lab 5: Windy Frozen Lake
# Nondeterministic World
import gym
import numpy as np
import matplotlib.pyplot as plt

def main():
    env = gym.make('FrozenLake-v0')

    # Initialize table with all zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n]) # the size of space in the game, the number of actions that it can choose

    # Learning paramter
    num_episodes = 2000
    dis = .99
    learning_rate = 0.9

    # Create lists to contain total rewards and steps per episode
    rList = []
    for i in range(num_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        rAll = 0
        done = False

        #e = 1.  / ((i // 100) + 1)
        
        # The Q-Table learning algorithm
        while not done:
            # Choose an action by greedily (with noise) picking from Q table
            action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

            # Choose an action by e-greedy
            #if np.random.rand(1) < e:
            #    action = env.action_space.sample()
            #else:
            #    action = np.argmax(Q[state, :])

            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(action)

            # Update Q-Table with new knowledge using learning rate
            Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + dis * np.max(Q[new_state, :]))

            rAll += reward
            state = new_state

        rList.append(rAll)

    print('Success rate: ' + str(sum(rList) / num_episodes))
    print('Final Q-Talbe Values')
    print(Q)
    plt.bar(range(len(rList)), rList, color='blue')
    plt.show()

if __name__ == '__main__':
    main()
